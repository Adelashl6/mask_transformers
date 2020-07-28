import os
import argparse
import time


def get_args():
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", type=str, default="Vatex", choices=['Vatex'],
                        help="Name of the dataset, will affect data loader, evaluation, etc")

    # model config
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, help="number of words in the vocabulary")
    parser.add_argument("--word_vec_size", type=int, default=768)
    parser.add_argument("--video_feature_size", type=int, default=1024, help="2048 appearance + 1024 flow")
    parser.add_argument("--max_vid_len", type=int, default=32, help="max length of video feature")
    parser.add_argument("--max_cap_len", type=int, default=30+2,
                        help="max length of text (sentence or paragraph), 30 for Vatex")
    parser.add_argument("--min_cap_len", type=int, default=2,
                        help="max length of text (sentence or paragraph), 30 for Vatex")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--num_encoder_hidden_layers", type=int, default=2, help="number of encoder transformer layers")
    parser.add_argument("--num_decoder_hidden_layers", type=int, default=2, help="number of decoder transformer layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--glove_path", type=str, default=None, help="extracted GloVe vectors")
    parser.add_argument("--freeze_glove", action="store_true", help="do not train GloVe vectors")
    parser.add_argument("--share_wd_cls_weight", action="store_true",
                        help="share weight matrix of the word embedding with the final classifier, ")

    parser.add_argument("--untied", action="store_true", help="Run untied model")
    parser.add_argument("--mtrans", action="store_true", help="Masked transformer model for single sentence generation")

    # training config -- learning rate
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--grad_clip", type=float, default=1, help="clip gradient, -1 == disable")
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="Use exponential moving average at training, float in (0, 1) and -1: do not use.  "
                             "ema_param = new_param * ema_decay + (1-ema_decay) * last_param")
    parser.add_argument("--BertAdam", action="store_true", help="AdamW")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Use soft target instead of one-hot hard target")
    parser.add_argument("--n_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_es_cnt", type=int, default=10,
                        help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("--batch_size", type=int, default=256, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=50, help="inference batch size")

    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("-block_ngram_repeat", type=int, default=0, help="block repetition of ngrams during decoding.")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")

    # data dir
    parser.add_argument("--ann_path", type=str, default='./data/Vatex', help='Path of HowTo100 Dataset Annotations.')
    parser.add_argument('--cap_file', type=str, default='vatex_captioning.pkl', help='Caption files.')
    parser.add_argument('--vocab_file', type=str, default='vatex_vocab.pkl', help='Vocabulary diction files.')
    parser.add_argument('--splits_file',type=str,default='vatex_splits.pkl', help='Splits files.')
    parser.add_argument('--glovev', type=str, default='vatex_vocab_glove.pt', help='glove word embedding.')
    # others
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Don't use pin_memory=True for dataloader. "
                             "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
    parser.add_argument("---num_workers", type=int, default=0,
                        help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--exp_id", type=str, default="res", help="id of the current run")
    parser.add_argument("--res_root_dir", type=str, default="results", help="dir to containing all the results")
    parser.add_argument("--save_model", default="model")
    parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="all",
                        help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("--seed", default=2019, type=int)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--eval_tool_dir", type=str, default="./standalone_eval")

    opt = parser.parse_args()
    assert not (opt.untied and opt.mtrans), "cannot be True for both"

    # single sentence
    if opt.untied:
        model_type = "untied_single"
    elif opt.mtrans:
        model_type = "mtrans_single"

    # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join([opt.dset_name, model_type, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")]))
    if opt.debug:
        opt.res_dir = "debug_" + opt.res_dir

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"
    return opt