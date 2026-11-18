# BPE Tokenizer and NER

A byte-pair encoding (BPE) tokenizer written from scratch in Python, trained on three text domains, and used as the input tokenization for a BiLSTM named-entity tagger.

Built together with Tali Lukatski.

## What it does

1. **BPE tokenizer** (`my_bpe_tokenizer.py`): learns merges over UTF-8 bytes, marks word starts with a space-prefixed token, and encodes/decodes text.
2. **Training** (`train_tokenizer.py`, `generate_tokenizers.py`): trains one tokenizer per domain, plus a tokenizer on the merged domains.
3. **NER tagger** (`train_ner_model.py`): a BiLSTM tagger over subword tokens. Only the first subtoken of each word receives the word's label; the rest are labeled as non-entity.

## Files

| File | Description |
| --- | --- |
| `base_tokenizer.py` | Abstract tokenizer interface (train, encode, decode, save, load). |
| `my_bpe_tokenizer.py` | BPE implementation. |
| `train_tokenizer.py` | Train and save one tokenizer from a text file. |
| `generate_tokenizers.py` | Train the three domain tokenizers. |
| `train_ner_model.py` | Train and evaluate the NER tagger on a tokenizer. |

## Usage

Expected inputs: plain-text domain files (one text per line) and tagged NER files. The data is not included in this repository.

```bash
pip install torch numpy tqdm regex
python3 train_tokenizer.py --domain_file data/domain_1.txt --output_dir tokenizers --vocab_size 5000
python3 train_ner_model.py --tokenizer_path tokenizers/tokenizer.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged
```

## Results

Measured on a private evaluation set (not included in this repository):

- Encoding speed: about 280,000 to 314,000 tokens per second.
- Reconstruction: 100% of the test texts decode back to the original text, ignoring whitespace.
- NER: dev F1 of 0.96 on one tokenizer and dataset setup.
