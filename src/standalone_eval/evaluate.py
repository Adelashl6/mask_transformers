import argparse
import time
import json
import sys
#sys.path.insert(0, './coco-caption')

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
import numpy as np


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


class VatexCaptionEval:
    """
    ground_truth_path: str, .jsonl file path to the ground truth captions
        Example line in the ground_truth_path file:
        {
             "vid_name": "friends_s08e08_seg01_clip_00",
             "duration": 61.03,
             "ts": [5.8, 8.24],
             "clip_id": 86618,
             "descs": [
                 {"desc": "Rachael walks up to Phoebe and Phoebe turns around.",
                  "type": "v",
                  "from_retrieval": false,
                  "desc_id": 109026
                  },
                  ...  # (ground-truth will have 4 such entries)
             ]
        }
    prediction_path: str, .jsonl file path to the generated captions
        Example line in the ground_truth_path file: (same structure as ground_truth but many entries are missing)
        {
             "clip_id": 86618,
             "descs": [
                {"desc": "Rachael walks up to Phoebe and Phoebe turns around."}
             ]  # if multiple descriptions are given, only use the first one in the list.
        }
    """
    def __init__(self, prediction_path):
        self.ground_truth = self.load_captions(prediction_path, is_ground_truth=True)
        self.prediction = self.load_captions(prediction_path, is_ground_truth=False)
        self.eval_res = {}
        self.eval_res_by_clip = {}  # TODO add eval res by clip

    @classmethod
    def load_captions(cls, filename, is_ground_truth=False):
        captions = load_jsonl(filename)
        if is_ground_truth:
            return {c["video_id"]: [{"caption": remove_nonascii(e)} for e in c["gt_captions"]] for c in captions}
        else:
            return {c["video_id"]: [{"caption": remove_nonascii(c["sentence"])}] for c in captions}

    def evaluate(self):
        # =================================================
        # Tokenization
        # =================================================
        print("Tokenization")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.ground_truth)
        preds = tokenizer.tokenize(self.prediction)

        # =================================================
        # Setup scorers
        # =================================================
        print("Setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print("Computing {} score...".format(scorer.method()))
            score, scores = scorer.compute_score(gts, preds)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.eval_res[m] = sc * 100
            else:
                self.eval_res[method] = score * 100


def start_eval(submission_path, output):
    evaluator = VatexCaptionEval(submission_path)
    evaluator.evaluate()
    print("Evaluation results: \n{}".format(json.dumps(evaluator.eval_res, indent=4)))
    save_json(evaluator.eval_res, output, save_pretty=True)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submission_path", type=str, help="Path to the submission (prediction) file")
    # parser.add_argument("-r", "--reference_path", type=str, help="Path to the reference (ground truth) file")
    parser.add_argument("-o", "--output", type=str, help="Path to save the metric results")
    args = parser.parse_args()
    start_eval(args.submission_path, args.output)
    print("Elapsed time {} seconds".format(time.time() - start))
