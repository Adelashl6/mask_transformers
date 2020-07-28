""" This module will handle the text generation with beam search. """

import torch
import copy
import torch.nn.functional as F

from src.rtransformer.model import NonRecurTransformerUntied
from src.rtransformer.masked_transformer import MTransformer
from src.rtransformer.beam_search import BeamSearch
from src.rtransformer.vatex_dataset import VatexDataset
import logging
logger = logging.getLogger(__name__)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=VatexDataset.EOS, pad_token_id=VatexDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda")

        self.model_config = checkpoint["model_cfg"]
        self.max_t_len = self.opt.max_cap_len
        self.max_v_len = self.opt.max_vid_len
        self.num_encoder_hidden_layers = self.opt.num_encoder_hidden_layers
        self.num_decoder_hidden_layers = self.opt.num_decoder_hidden_layers
        if model is None:
            if opt.untied:
                logger.info("Use untied non-recurrent single sentence model")
                model = NonRecurTransformerUntied(self.model_config).to(self.device)
            elif opt.mtrans:
                logger.info("Use masked transformer -- another non-recurrent single sentence model")
                model = MTransformer(self.model_config).to(self.device)
            # model = RecursiveTransformer(self.model_config).to(self.device)
        model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

        # self.eval_dataset = eval_dataset

    def translate_batch_beam(self, video_features, video_masks, text_input_ids, text_masks, text_input_labels, model,
                             beam_size, n_best, min_length, max_length, block_ngram_repeat, exclusion_idxs,
                             device):
        base_beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=len(video_masks),
            pad=VatexDataset.PAD,
            eos=VatexDataset.EOS,
            bos=VatexDataset.BOS,
            min_length=min_length,
            max_length=max_length,
            mb_device=device,
            block_ngram_repeat=block_ngram_repeat,
            exclusion_tokens=exclusion_idxs
        )

        def duplicate_for_beam(encoder_outputs, video_masks, beam_size):
            outputs = [tile(e, beam_size, dim=0) for e in encoder_outputs] \
                if encoder_outputs[0] is not None else [None] * len(encoder_outputs)
            masks = tile(video_masks, beam_size, dim=0)
            return outputs, masks

        encoder_outputs = model.encode(video_features, video_masks)  # (N, Lv, D)
        encoder_outputs, video_masks = duplicate_for_beam(encoder_outputs, video_masks, beam_size=beam_size)
        beam = copy.deepcopy(base_beam)
        model_inputs = dict(
            encoder_outputs=encoder_outputs,
            video_masks=video_masks
        )
        #beam = copy.deepcopy(base_beam)
        bsz = len(video_masks)
        text_input_ids = model_inputs['video_masks'].new_zeros(bsz, max_length).long()  # all zeros
        text_masks = model_inputs['video_masks'].new_zeros(bsz, max_length)
        #text_input_labels = text_input_labels.new_full((bsz, max_length), -1).long()
        encoder_outputs = model_inputs["encoder_outputs"]
        encoder_masks = video_masks

        for dec_idx in range(max_length):
            text_input_ids[:, dec_idx] = beam.current_predictions
            text_masks[:, dec_idx] = 1
            copied_encoder_outputs = copy.deepcopy(encoder_outputs)
            _, pred_scores = model.decode(
                text_input_ids, text_masks, None, copied_encoder_outputs, video_masks)

            pred_scores[:, VatexDataset.UNK] = -1e10  # remove `[UNK]` token
            logprobs = torch.log(F.softmax(pred_scores[:, dec_idx], dim=1))  # (N * beam_size, vocab_size)
            beam.advance(logprobs)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            if any_beam_is_finished:
                # update input args
                select_indices = beam.current_origin  # N * B, i.e. batch_size * beam_size
                text_input_ids = text_input_ids.index_select(0, select_indices)
                text_masks = text_masks.index_select(0, select_indices)
                #text_input_labels = text_input_labels.index_select(0, select_indices)
                #encoder_outputs = encoder_outputs.index_select(0, select_indices)
                encoder_masks = encoder_masks.index_select(0, select_indices)
                if encoder_outputs[0] is None:
                    encoder_outputs = [None] * len(select_indices)
                else:
                    encoder_outputs = [e.index_select(0, select_indices) for e in encoder_outputs]

        # fill in generated words
        bsz = len(beam.predictions)
        text_input_ids = model_inputs["video_masks"].new_zeros(bsz, max_length).long()  # zeros
        text_masks = model_inputs["video_masks"].new_zeros(bsz, max_length)
        for batch_idx in range(len(beam.predictions)):
            cur_sen_ids = beam.predictions[batch_idx][0].cpu().tolist()  # use the top sentences
            cur_sen_len = len(cur_sen_ids)
            text_input_ids[batch_idx, :cur_sen_len] = text_input_ids.new(cur_sen_ids)
            text_masks[batch_idx, :cur_sen_len] = 1

        # compute memory, mimic the way memory is generated at training time
        text_input_ids, text_masks = mask_tokens_after_eos(text_input_ids, text_masks)
        return text_input_ids

    @classmethod
    def translate_batch_single_sentence_untied_greedy(
            cls, video_features, video_masks, text_input_ids, text_masks, text_input_labels,
            model, start_idx=VatexDataset.BOS, unk_idx=VatexDataset.UNK):
        """The first few args are the same to the input to the forward_step func
        Note:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """
        encoder_outputs = model.encode(video_features, video_masks)  # (N, Lv, D)

        config = model.config
        max_t_len = config.get('max_t_len')
        bsz = len(text_input_ids)
        text_input_ids = text_input_ids.new_zeros(text_input_ids.size())  # all zeros
        text_masks = text_masks.new_zeros(text_masks.size())
        # zero# all zeros
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_t_len):
            text_input_ids[:, dec_idx] = next_symbols
            text_masks[:, dec_idx] = 1
            _, pred_scores = model.decode(
                text_input_ids, text_masks, None, encoder_outputs, video_masks)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            # next_words = pred_scores.max(2)[1][:, dec_idx]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words
        text_input_ids, text_masks = mask_tokens_after_eos(text_input_ids, text_masks)
        return text_input_ids  # (N, Lt)

    def translate_batch(self, model_inputs, use_beam=False, untied=False, mtrans=False):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        if use_beam:
            with torch.no_grad():
                video_features, video_masks, text_input_ids, text_masks, text_input_labels = model_inputs
                return self.translate_batch_beam(video_features, video_masks, text_input_ids, text_masks, text_input_labels, self.model,
                                                 beam_size=self.opt.beam_size, n_best=self.opt.n_best,
                                                 min_length=self.opt.min_cap_len, max_length=self.opt.max_cap_len,
                                                 block_ngram_repeat=self.opt.block_ngram_repeat, exclusion_idxs=[],
                                                 device=self.device)

        else:# single sentence
            if untied or mtrans:
                with torch.no_grad():
                    video_features, video_masks, text_input_ids, text_masks, text_input_labels = model_inputs
                    return self.translate_batch_single_sentence_untied_greedy(video_features, video_masks, text_input_ids, text_masks, text_input_labels, self.model)

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids):
        """ replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = VatexDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = VatexDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks
