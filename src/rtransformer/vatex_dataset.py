import torch
import pickle
import os
import logging
import nltk
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class VatexDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"
    SEP = 4
    EOS = 3
    BOS = 2
    UNK = 1
    PAD = 0
    IGNORE = -1  # used to calculate loss

    def __init__(self, opt, mode='train'):
        super(VatexDataset, self).__init__()
        self.opt = opt
        self.mode = mode

        # Load the caption annotations
        self.caption_hub = pickle.load(open(os.path.join(opt.ann_path, opt.cap_file), 'rb'))

        # Read the training/testing split
        self.splits = pickle.load(open(os.path.join(opt.ann_path, opt.splits_file), 'rb'))[mode]


        # Read the vocabulary file
        vocab = pickle.load(open(os.path.join(opt.ann_path, opt.vocab_file), 'rb'))

        # Load the vocabulary dictionary
        self.word2idx = vocab['word2idx']
        self.idx2word = vocab['idx2word']
        self.data = self._load_data()
        self.max_v_len = opt.max_vid_len
        self.max_cap_len = opt.max_cap_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, meta = self.convert_example_to_features(self.data[index])
        return data, meta

    def convert_example_to_features(self, example):
        """
        {"vid_name": str,
         "duration": float,
         "ts": [st(float), ed(float)],
         "desc": str,
         "clip_id": str
        }
        """

        video_feat, video_masks = self.get_video_feature(example)
        assert video_feat.shape[0] == 32
        if self.mode == 'val':
            text_input_ids = [0]*self.opt.max_cap_len
            text_masks = [0]*self.opt.max_cap_len
            text_labels = [-1]*self.opt.max_cap_len
        else:
            text_input_ids, text_masks = self.get_caption(example)
            # shifted right, `-1` is ignored when calculating CrossEntropy Loss
            text_labels = [self.IGNORE if m == 0 else tid
                              for tid, m in zip(text_input_ids, text_masks)][1:] + [self.IGNORE]

        data = dict(
            text_ids=np.array(text_input_ids).astype(np.int64),
            text_masks=np.array(text_masks).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feat.astype(np.float32),
            video_masks=np.array(video_masks).astype(np.float32)

        )
        meta = example
        return data, meta


    def _load_data(self):
        data = []
        for idx in range(len(self.splits)):
            video_id = self.splits[idx]
            # Load the caption
            annotation = self.caption_hub[video_id]
            captions = annotation['raw_captions']
            if self.mode == 'train':
                sample = np.random.randint(10)
                caption = captions[sample]
            else:
                caption = captions
            feature = annotation['feature'][0]
            data.append(dict(video_feat=torch.tensor(feature),
                             captions=caption,
                             video_id=video_id,
                             arr_length=feature.shape[0]))
        logging.info("Loading complete! {} captions".format(len(data)))
        return data

    def _tokenize_and_pad_sentence(self, sentence, max_sen_l):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_cap_len
        All non-PAD values are valid, with a mask value of 1
        """
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_sen_l - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_sen_l - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_sen_l - valid_l)
        return sentence_tokens, mask

    def get_video_feature(self, example):
        raw_feat = example['video_feat']
        valid_l = example['arr_length']
        video_feat = np.zeros((self.max_v_len, raw_feat.shape[1]))
        video_feat[:valid_l] = raw_feat
        video_mask = [1]*valid_l + [0]*(self.max_v_len - valid_l)
        return video_feat, video_mask

    def get_caption(self, example):
        """example: """
        caption_tokens, caption_mask = self._tokenize_and_pad_sentence(example['captions'], max_sen_l=self.max_cap_len)
        caption_input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in caption_tokens]
        return caption_input_ids, caption_mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[str(wid)] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[str(wid)] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"video_id": e[1]["video_id"],
                   "gt_captions": e[1]["captions"],
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, batch_meta

