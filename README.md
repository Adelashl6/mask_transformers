The code is adapted from https://github.com/jayleicn/recurrent-transformer. In this code, we preprocess the recently proposed video captioning dataset VATEX and use the mask transformer as our baseline model for captioning task.

## Data:
The preprocessed data can be acessed through the link:
https://drive.google.com/drive/folders/1rp3oxvGHL0JsCgsq7TDgE8JjeSCCXxGo.
The folders contains four files listed below:
```
1. **vatex_captioning.pkl** contains ground-truth captions for each video in Vatex.
2. **vatex_splits.pkl** includes the name lists of training and testing set.
3. **vatex_vocab.pkl** are the vocabulary of words in ground-truth captions.
4. **vatex_vocab_glove.pt** stores the glove embeddings of all the words in the vocabulary.
```

Downloads the above files and move them to ./src/data/Vatex.

# Performance:
| METEOR| BLEU@4  | CIDER  | ROUGE-L |
| :-:   | :-: | :-: | :-: |
| 23.0 | 28.03 | 50.45 | 47.01 |

