"""
This script handles the training process.
"""


import math
import time
import random
import numpy as np
import os
import json
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.opt import get_args
from src.rtransformer.vatex_dataset import single_sentence_collate, prepare_batch_inputs
from src.rtransformer.vatex_dataset import VatexDataset
from src.rtransformer.model import NonRecurTransformerUntied
from src.rtransformer.masked_transformer import MTransformer
from src.rtransformer.optimization import BertAdam, EMA, ScheduledOptim
import torch.optim as optim
from src.translator import Translator
from src.translate import run_translate
from src.utils import save_parsed_args_to_json, save_jsonl, load_json, count_parameters, merge_dicts
from src.standalone_eval.evaluate import start_eval
import logging
logger = logging.getLogger(__name__)


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(VatexDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


def train_epoch(model, training_data, optimizer, ema, device, opt,  epoch):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in enumerate(training_data):
        niter = epoch * len(training_data) + batch_idx
        print("Train/LearningRate", float(optimizer._optimizer.param_groups[0]["lr"]), niter)

        # single sentence
        if opt.untied or opt.mtrans:
                # prepare data
            meta = batch[1]
            batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)
            video_feature = batched_data["video_feature"]
            video_mask = batched_data["video_masks"]
            text_ids = batched_data["text_ids"]
            text_mask = batched_data["text_masks"]
            text_labels = batched_data["text_labels"]

            if opt.debug:
                def print_info(cur_data, batch_idx):
                    logger.info("text_ids \n{}".format(cur_data["text_ids"][batch_idx]))
                    logger.info("text_masks \n{}".format(cur_data["text_masks"][batch_idx]))
                    logger.info("text_labels \n{}".format(cur_data["text_labels"][batch_idx]))

                print_info(batched_data, 0)

        # forward & backward
        optimizer.zero_grad()
        loss, pred_scores = model(video_feature, video_mask, text_ids, text_mask, text_labels)

        # make it consistent with other configs
        pred_scores_list = [pred_scores]
        input_labels_list = [text_labels]

        loss.backward()
        if opt.grad_clip != -1:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        #optimizer.step()
        optimizer.step_and_update_lr()

        # update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels_list):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(VatexDataset.IGNORE)
            n_word += valid_label_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        if opt.debug:
            break
    torch.autograd.set_detect_anomaly(False)

    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_language_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode="val"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
        0, run inference
        1, Get METEOR, BLEU1-4, CIDEr scores
        2, Get vocab size, sentence length
        """
    translator = Translator(opt, checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, opt=opt)
    res_filepath = os.path.abspath(opt.save_model + "_tmp_greedy_pred_{}.json".format(eval_mode))
    save_jsonl(json_res, res_filepath)
    # COCO language evaluation
    #reference_path = os.path.abspath(opt.reference_path)
    metrics_path = res_filepath.replace(".json", "_lang_metrics.json")
    start_eval(res_filepath, metrics_path)
    metrics = load_json(metrics_path)
    return metrics, [res_filepath, metrics_path]


def train(model, training_data, validation_data, device, opt):
    model = model.to(device)
    # Prepare optimizer
    if opt.BertAdam:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
    if opt.ema_decay != -1:
        ema = EMA(opt.ema_decay)
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema.register(name, p.data)
    else:
        ema = None

    num_train_optimization_steps = len(training_data) * opt.n_epoch
    if opt.BertAdam:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=opt.lr,
                             warmup=opt.lr_warmup_proportion,
                             t_total=num_train_optimization_steps,
                             schedule="warmup_linear")
    else:
        warm_up_steps = opt.lr_warmup_proportion * opt.n_epoch * len(training_data)
        optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                              betas=(0.9, 0.98), eps=1e-09), 768, warm_up_steps)

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        logger.info("Training performance will be written to file: {} and {}".format(
            log_train_file, log_valid_file))

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,METEOR,BLEU@4,CIDEr,re4\n")

    prev_best_score = 0.
    es_cnt = 0
    for epoch_i in range(opt.n_epoch):
        logger.info("[Epoch {}]".format(epoch_i))

        # schedule sampling prob update, TODO not implemented yet
        start = time.time()
        if ema is not None and epoch_i != 0:  # use normal parameters for training, not EMA model
            ema.resume(model)
        train_loss, train_acc = train_epoch(
            model, training_data, optimizer, ema, device, opt,  epoch_i)
        logger.info("[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=math.exp(min(train_loss, 100)), acc=100*train_acc, elapse=(time.time()-start)/60.))
        niter = (epoch_i + 1) * len(training_data)  # number of bart
        print("Train/Acc", train_acc, niter)
        print("Train/Loss", train_loss, niter)

        # Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # EMA model

        # Note here we use greedy generated words to predicted next words, the true inference situation.
        checkpoint = {
            "model": model.state_dict(),  # EMA model
            "model_cfg": model.config,
            "opt": opt,
            "epoch": epoch_i}

        val_greedy_output, filepaths = eval_language_metrics(
            checkpoint, validation_data, opt, eval_mode="val", model=model)
        cider = val_greedy_output["CIDEr"]
        bleu4 = val_greedy_output["Bleu_4"]
        meteor = 0   #val_greedy_output["METEOR"]  TODO there exsit some bugs, set this metric to 0 temporarily
        rouge = val_greedy_output["ROUGE_L"]
        logger.info("[Val] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} re4 {r:.2f}"
                    .format(m=meteor,
                            b=val_greedy_output["Bleu_4"],
                            c=val_greedy_output["CIDEr"],
                            r=val_greedy_output["ROUGE_L"]))
        print("Val/METEOR", meteor, niter)
        print("Val/Bleu_4", val_greedy_output["Bleu_4"], niter)
        print("Val/CIDEr", val_greedy_output["CIDEr"], niter)
        print("Val/ROUGE_L", val_greedy_output["ROUGE_L"], niter)

        if opt.save_mode == "all":
            model_name = opt.save_model + "_e{e}_b{b}_m{m}_c{c}_r{r}.chkpt".format(
                e=epoch_i, b=round(bleu4, 2), m=round(meteor, 2),
                c=round(cider, 2), r=round(rouge, 2))
            torch.save(checkpoint, model_name)
        elif opt.save_mode == "best":
            model_name = opt.save_model + ".chkpt"
            if cider > prev_best_score:
                es_cnt = 0
                prev_best_score = cider
                torch.save(checkpoint, model_name)
                new_filepaths = [e.replace("tmp", "best") for e in filepaths]
                for src, tgt in zip(filepaths, new_filepaths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if es_cnt > opt.max_es_cnt:  # early stop
                    logger.info("Early stop at {} with CIDEr {}".format(epoch_i, prev_best_score))
                    break
        cfg_name = opt.save_model + ".cfg.json"
        save_parsed_args_to_json(opt, cfg_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:
                log_tf.write("{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n".format(
                    epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))
                log_vf.write("{epoch},{m:.2f},{b:.2f},{c:.2f},{r:.2f}\n".format(
                    epoch=epoch_i,
                    m=meteor,
                    b=val_greedy_output["Bleu_4"],
                    c=val_greedy_output["CIDEr"],
                    r=val_greedy_output["ROUGE_L"]))

        if opt.debug:
            break

    #writer.close()


def main():
    opt = get_args()
    print(opt.mtrans)
    print(opt.untied)
    print(opt.share_wd_cls_weight)
    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_dataset = VatexDataset(opt, mode="train")
    # add 10 at max_n_sen to make the inference stage use all the segments
    val_dataset = VatexDataset(opt, mode="val")

    collate_fn = single_sentence_collate
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    opt.vocab_size = len(train_dataset.word2idx)
    print(json.dumps(vars(opt), indent=4, sort_keys=True))

    device = torch.device("cuda")

    rt_config = dict(
        hidden_size=opt.hidden_size,
        intermediate_size=opt.intermediate_size,  # after each self attention
        vocab_size=opt.vocab_size,  # get from word2idx
        word_vec_size=opt.word_vec_size,
        video_feature_size=opt.video_feature_size,
        max_position_embeddings=opt.max_cap_len,  # get from max_seq_len
        max_v_len=opt.max_vid_len,  # max length of the videos
        max_t_len=opt.max_cap_len,  # max length of the text
        layer_norm_eps=opt.layer_norm_eps,  # bert layernorm
        hidden_dropout_prob=opt.hidden_dropout_prob,  # applies everywhere except attention
        num_encoder_hidden_layers=opt.num_encoder_hidden_layers,# number of encoder transformer layers
        num_decoder_hidden_layers=opt.num_decoder_hidden_layers,# number of decoder transformer layers
        num_attention_heads=opt.num_attention_heads,
        attention_probs_dropout_prob=opt.attention_probs_dropout_prob,  # applies only to self attention
        initializer_range=opt.initializer_range,
        label_smoothing=opt.label_smoothing,
        share_wd_cls_weight=opt.share_wd_cls_weight
    )

     # single sentence, including untied
    if opt.untied:
        logger.info("Use untied non-recurrent single sentence model")
        model = NonRecurTransformerUntied(rt_config)
    elif opt.mtrans:
        logger.info("Use masked transformer -- another non-recurrent single sentence model")
        model = MTransformer(rt_config)

    if opt.glove_path is not None:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    count_parameters(model)
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        count_parameters(model.embeddings.word_embeddings)

    train(model, train_loader, val_loader, device, opt)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

