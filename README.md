The code is adapted from https://github.com/jayleicn/recurrent-transformer. In this code, we preprocess the recently proposed video captioning dataset VATEX and use the mask transformer as our baseline model for captioning task.

## Data:
The data can be acessed through the link:
https://drive.google.com/drive/folders/1rp3oxvGHL0JsCgsq7TDgE8JjeSCCXxGo.
Downloads the files and move them to ./src/data/Vatex.
The folder should contain four files:
```
1. vatex_captioning.pkl
2. vatex_splits.pkl
3. vatex_vocab.pkl
4. vatex_vocab_glove.pt
```

# Performance:
METEOR: 23.0 
BLEU@4: 28.03
CIDER: 50.45 
ROUGE-L: 47.01
