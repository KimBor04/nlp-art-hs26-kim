# NLP Project 1 – Abductive Commonsense Reasoning

HSLU | Natural Language Processing | HS26

## 1. Project Overview

This project focuses on **Abductive Commonsense Reasoning** using the ART (Abductive Reasoning in Narrative Text) dataset.

The goal is to select the most plausible explanation connecting two observations in a short story.

Each example contains:

- **Observation 1 (O1):** The beginning of a story.
- **Observation 2 (O2):** The ending of the story.
- **Hypothesis 1 (H1):** First possible explanation.
- **Hypothesis 2 (H2):** Second possible explanation.
- **Label:** Indicates the more plausible hypothesis.

Two neural network architectures are required:

1. **Architecture 1:** Pretrained fastText word embeddings with a two-layer MLP classifier.
2. **Architecture 2:** The same pretrained embeddings with a two-layer RNN (LSTM or GRU) and a two-layer classifier.

The embeddings remain frozen in both architectures.

Experiments are tracked using Weights & Biases (W&B).

## 2. Dataset

The dataset is provided by HSLU through Hugging Face:

https://huggingface.co/datasets/amarfurt/art-nlp-hs26

| Split | Examples | Purpose |
|---|---:|---|
| Training | 16,801 | Model training and vocabulary construction |
| Validation | 1,000 | Model selection and hyperparameter tuning |
| Test | 1,532 | Final evaluation |

Load the dataset using:

```python
from datasets import load_dataset

dataset = load_dataset("amarfurt/art-nlp-hs26")
```

The vocabulary and preprocessing statistics are derived from the training split.

The validation split is reserved for model selection. The test split is not used for model fitting or hyperparameter tuning.

## 3. Preprocessing

The preprocessing pipeline is implemented in `src/preprocessing.py` and explored in `notebooks/01_data_exploration.ipynb`.

### Text Cleaning

The training data was checked for HTML tags, URLs, HTML entities, emojis, invisible characters, and unusual Unicode symbols.

No HTML tags, URLs, emojis, HTML entities, or invisible characters were found. Only a few corrupted or unusual characters were identified.

Apostrophe variants are normalized, and clearly identifiable corrupted apostrophes are repaired. Other corrupted characters remain unchanged.

### Tokenization Decisions

| Step | Decision |
|---|---|
| Tokenization | Regex-based word-level tokenizer |
| Lowercasing | Yes |
| Stopword removal | No |
| Punctuation removal | No |
| Stemming | No |
| Lemmatization | No |
| Contractions | Expand negations and unambiguous forms |
| Possessives | Represent most `'s` forms using a separate `possessive` token |
| Rare words | Keep if covered by pretrained fastText |
| Unknown words | Map to `[UNK]` |
| Truncation | None |

Negations and stopwords are preserved because small words such as "not", "before", and "after" can change the meaning of a story.

Ambiguous contractions such as `he'd` and `it's` are not automatically expanded because their meaning depends on context.

### Input Format

The four text fields are concatenated into one sequence:

```text
O1 [SEP] O2 [SEP] H1 [SEP] H2
```

Three `[SEP]` tokens separate the text segments.

The original labels `{1, 2}` are converted to `{0, 1}` for classification.

## 4. Word Embeddings

Pretrained fastText embeddings are loaded through Gensim.

**Model:** `fasttext-wiki-news-subwords-300`

**Embedding dimensions:** 300

The vocabulary is constructed using training data only. Words are included when a pretrained fastText vector is available.

### Vocabulary Coverage

| Metric | Result |
|---|---:|
| Vocabulary size (including special tokens) | 15,038 |
| Training-derived vocabulary words | 15,035 |
| Vocabulary coverage | 95.03% |
| Total training tokens | 656,675 |
| Known training tokens | 654,159 |
| Unknown training tokens | 2,516 |
| Token coverage | 99.62% |

### Special Tokens

| Token | ID | Initialization |
|---|---:|---|
| `[PAD]` | 0 | Zero vector |
| `[UNK]` | 1 | Mean of available pretrained word vectors |
| `[SEP]` | 2 | Fixed random vector |

The embedding matrix has shape `(15038, 300)`.

All embedding weights are frozen during training.

**fastText limitation:** Although the pretrained model is based on fastText, the implementation uses fixed vocabulary lookup rather than full character n-gram inference for unseen words.

Words outside the constructed vocabulary map to `[UNK]`, even if the original fastText approach could represent them using subword information.

## 5. Data Loading and Padding

The DataLoader implementation is located in `src/data_loader.py`.

Sequences are encoded and dynamically padded to the longest sequence within each batch.

Current batch size: **32**

Each batch contains:

- `input_ids`: Padded token IDs.
- `lengths`: Original sequence lengths.
- `labels`: Classification labels.

The original lengths allow padding masks to be reconstructed and will support length-aware RNN processing.

### Verification Results

| Metric | Training | Validation |
|---|---:|---:|
| Examples | 16,801 | 1,000 |
| Minimum length | 20 | 23 |
| Mean length | 42.09 | 42.56 |
| 95th percentile | 55 | 55 |
| Maximum length | 92 | 93 |
| Padding percentage | 31.89% | 33.09% |

The following checks passed:

- Dynamic padding works correctly.
- Padding positions contain `[PAD]`.
- Original sequence lengths are preserved.
- Padding masks reconstructed from lengths are correct.
- No examples are lost or sequences truncated.

Model-level handling of padding is verified separately from the DataLoader checks.

## 6. Model Architectures

### Architecture 1 – fastText + MLP

The baseline uses pretrained fastText embeddings and a two-layer classifier with ReLU activation.

Mean pooling creates a fixed-size text representation for classification.

Only the classifier is trained; the pretrained embeddings remain frozen.

An important limitation of mean pooling is that word order is not preserved.

The baseline implementation is part of the current project. Final results using the updated preprocessing pipeline are not yet documented here.

### Architecture 2 – fastText + RNN

The second architecture will use:

- The same frozen fastText embeddings.
- A two-layer GRU or LSTM.
- A two-layer classifier with ReLU activation.

Unlike mean pooling, an RNN can model sequential dependencies and word order.

**Status:** Not yet completed.

## 7. Expected Results

The original predictions are documented in `EXPECTATIONS.md`.

| Model | Expected test accuracy |
|---|---:|
| Random baseline | 50% |
| Architecture 1 – MLP | 68% |
| Architecture 2 – RNN | 75% |

These are predictions, **not measured results**.

The RNN is expected to outperform the MLP because it can capture sequential relationships. However, the more complex architecture may also be more prone to overfitting.

The original expectations will remain unchanged.

## 8. Verification and Reproducibility

### Preprocessing Consistency

The notebook tokenizer and reusable preprocessing module were compared across all training text fields.

Results:

- Tokenizer mismatches: 0
- Vocabulary comparison: Identical
- Encoded example comparison: Identical

### Vocabulary Leakage Audit

All four checks passed:

- All vocabulary words originate from training data.
- All included vocabulary words have pretrained fastText vectors.
- Vocabulary reconstruction produces identical token IDs.
- All eligible training words are included.

This verifies the vocabulary-construction stage. Training-loop leakage and model-selection behavior require separate checks.

### Reproducing Preprocessing

Required Python libraries include:

- PyTorch
- Hugging Face Datasets
- Gensim
- NumPy
- pandas

Open and execute:

`project1/notebooks/01_data_exploration.ipynb`

The notebook explores the dataset, loads fastText embeddings, constructs the vocabulary, verifies preprocessing and creates the DataLoaders.

The vocabulary and embedding weights are saved to:

`project1/artifacts/fasttext_art.pt`

Generated artifacts should be excluded from Git because they can be regenerated.

## 9. Experiment Tracking

Weights & Biases is used to track model experiments.

Planned tracked metrics include:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

Upcoming experiments will compare learning rates, hidden dimensions, batch sizes and model architectures.

Final model comparisons and the saved W&B workspace link will be added after training.