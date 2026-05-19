# DA6401 - Assignment 3: Transformer for Machine Translation

Implementation of the "Attention Is All You Need" Transformer from
scratch in PyTorch, trained for German to English translation on the
Multi30k dataset.

## Links

- GitHub repo: https://github.com/Soham-Shah-20072004/da6401_assignment_3_myfork
- W&B report: https://api.wandb.ai/links/me22b191-indian-institute-of-technology-madras/vv57313g

## Files

```text
model.py          # Transformer: attention, encoder/decoder, masks, infer()
lr_scheduler.py   # Noam learning-rate scheduler
dataset.py        # Multi30k loading, spacy tokenization, vocab, batching
train.py          # label smoothing, training loop, greedy decode, BLEU
experiments.py    # report runs for sections 2.1-2.5
requirements.txt
```

## Usage

Install deps and the spacy languages, then train:

```bash
pip install -r requirements.txt
pip install gdown
python -c "from experiments import run_all; run_all()"
```

`run_all()` trains the baseline plus the five ablations and logs
everything (curves, attention heatmaps, BLEU) to W&B. A single
experiment can be run with `run_one("baseline")`.

For evaluation, `Transformer()` (no args) downloads the trained
checkpoint from Drive inside `__init__` and `model.infer(german_sentence)`
returns the English translation.
