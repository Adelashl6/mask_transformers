""" Translate input text with trained model. """

import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
import numpy as np
from src.translator import Translator
from src.rtransformer.vatex_dataset import VatexDataset, single_sentence_collate, prepare_batch_inputs
from src.utils import load_json, merge_dicts, save_json, save_jsonl
from src.standalone_eval.evaluate import start_eval

'''
def sort_res(res_dict):
    """res_dict: the submission json entry `results`"""
    final_res_dict = {}
    for k, v in res_dict.items():
        final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
    return final_res_dict
'''


def run_translate(eval_data_loader, translator, opt):
    # submission template

    batch_res = []
    for raw_batch in tqdm(eval_data_loader, mininterval=2, desc="  - (Translate)"):
        meta = raw_batch[1]
        batched_data = prepare_batch_inputs(raw_batch[0], device=translator.device)
        if opt.untied or opt.mtrans:
            model_inputs = [
                batched_data["video_feature"],
                batched_data["video_masks"],
                batched_data["text_ids"],
                batched_data["text_masks"],
                batched_data["text_labels"]
                ]
        dec_seq = translator.translate_batch(
            model_inputs, use_beam=opt.use_beam, untied=opt.untied, mtrans=opt.mtrans)

            # example_idx indicates which example is in the batch
        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, meta)):
            cur_data = {
                "video_id": cur_meta["video_id"],
                "sentence": eval_data_loader.dataset.convert_ids_to_sentence(
                    cur_gen_sen.cpu().tolist()),
                "gt_captions": cur_meta["gt_captions"]
                }
            batch_res.append(cur_data)

        if opt.debug:
            break

    #batch_res["results"] = sort_res(batch_res["results"])
    return batch_res


def get_data_loader(opt, eval_mode="val"):
    eval_dataset = VatexDataset(opt, mode=eval_mode)
    collate_fn = single_sentence_collate
    eval_data_loader = DataLoader(eval_dataset, collate_fn=collate_fn,
                                  batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return eval_data_loader


def main():
    parser = argparse.ArgumentParser(description="translate.py")

    parser.add_argument("-eval_split_name", choices=["val", "test_public"])
    parser.add_argument("-eval_path", type=str, help="Path to eval data")
    parser.add_argument("-res_dir", required=True, help="path to dir containing model .pt file")
    parser.add_argument("-batch_size", type=int, default=50, help="batch size")

    # beam search configs
    parser.add_argument("-use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("-beam_size", type=int, default=3, help="beam size")
    parser.add_argument("-n_best", type=int, default=1, help="stop searching when get n_best from beam search")
    parser.add_argument("-min_sen_len", type=int, default=2, help="minimum length of the decoded sentences")
    parser.add_argument("-max_sen_len", type=int, default=30+2, help="maximum length of the decoded sentences")
    parser.add_argument("-block_ngram_repeat", type=int, default=0, help="block repetition of ngrams during decoding.")
    parser.add_argument("-length_penalty_name", default="none",
                        choices=["none", "wu", "avg"], help="length penalty to use.")
    parser.add_argument("-length_penalty_alpha", type=float, default=0.,
                        help="Google NMT length penalty parameter (higher = longer generation)")
    parser.add_argument("-num_workers", type=int, default=0,
                        help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-seed", default=2019, type=int)
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-mtrans", action="store_true")
    parser.add_argument("-untied", action="store_true")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    print(opt.use_beam)
    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    checkpoint = torch.load(os.path.join(opt.res_dir, "model.ckpt"))

    decoding_strategy = "beam{}_lp_{}_la_{}".format(
        opt.beam_size, opt.length_penalty_name, opt.length_penalty_alpha) if opt.use_beam else "greedy"
    save_json(vars(opt), os.path.join(opt.res_dir, "{}_eval_cfg.json".format(decoding_strategy)), save_pretty=True)

    # add some of the train configs
    train_opt = checkpoint["opt"]  # EDict(load_json(os.path.join(opt.res_dir, "model.cfg.json")))
    for k in train_opt.__dict__:
        if k not in opt.__dict__:
            setattr(opt, k, getattr(train_opt, k))

    eval_data_loader = get_data_loader(opt)

    # setup model
    translator = Translator(opt, checkpoint)

    pred_file = os.path.join(opt.res_dir, "{}_pred_{}.jsonl".format(decoding_strategy, opt.eval_split_name))
    pred_file = os.path.abspath(pred_file)
    if not os.path.exists(pred_file):
        json_res = run_translate(eval_data_loader, translator, opt=opt)
        save_jsonl(json_res, pred_file)
    else:
        print("Using existing prediction file at {}".format(pred_file))
    metrics_path = pred_file.replace(".json", "_lang_metrics.json")
    start_eval(pred_file, metrics_path)
    print("[Info] Finished {}.".format(opt.eval_split_name))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

