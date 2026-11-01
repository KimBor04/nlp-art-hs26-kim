# NLP Project 1 — Abductive Commonsense Reasoning

**HSLU · Natural Language Processing · HS26**

This project investigates **abductive commonsense reasoning** on the ART (Abductive Reasoning in Narrative Text) dataset. Given the beginning and end of a short story, a model must choose which of two candidate hypotheses provides the more plausible explanation.

We implement the two required architectures—**fastText + MLP** and **fastText + two-layer GRU**—and investigate an additional **PairwiseGRU** with shared hypothesis scoring. All models use **frozen, pretrained 300-dimensional fastText word embeddings**. Training, tuning, evaluation, and robustness experiments are documented in seven Jupyter notebooks and tracked using Weights & Biases (W&B).

## Key results

The following are the **original, validation-selected seed-42 checkpoints**, evaluated on the official test set (1,532 examples):

| Architecture | Validation accuracy | Test accuracy | Test macro-F1 | Validation-to-test drop |
|---|---:|---:|---:|---:|
| fastText + MLP | 65.00% | 51.57% | 0.5154 | 13.43 pp |
| fastText + Standard GRU | 68.50% | **54.05%** | **0.5404** | 14.45 pp |
| fastText + PairwiseGRU | **71.10%** | 53.98% | 0.5398 | 17.12 pp |

**Main finding:** PairwiseGRU performs best on validation data and enforces hypothesis-order symmetry, but its advantage does not transfer to the official test set. The Standard GRU achieves the highest original test accuracy (**54.05%**), only slightly above the approximately 50% chance baseline. The large validation-to-test drops motivate the dataset-shift and generalization analyses below.

## 1. Task and dataset

Each ART example contains:

- **O1:** The first observation (story beginning).
- **O2:** The second observation (story ending).
- **H1 / H2:** Two candidate explanations connecting the observations.
- **Label:** The more plausible hypothesis (1 or 2 in the source data).

**Dataset:** [amarfurt/art-nlp-hs26 on Hugging Face](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)

| Official split | Examples | Primary purpose |
|---|---:|---|
| Train | 16,801 | Model fitting and initial vocabulary construction |
| Validation | 1,000 | Hyperparameter tuning and original checkpoint selection |
| Test | 1,532 | Original final evaluation; subsequent read-only and exploratory analyses |

```python
from datasets import load_dataset

dataset = load_dataset("amarfurt/art-nlp-hs26")
```

Source labels `{1, 2}` are mapped to `{0, 1}` for PyTorch classification. The original test checkpoints were fixed using validation results before their first test evaluation in Notebook 06. Later experiments that also inspect the test set are explicitly treated as **exploratory**, not as a new independent final evaluation.

## 2. Preprocessing and embeddings

Reusable preprocessing is implemented in `project1/src/preprocessing.py` and explored in `project1/notebooks/01_data_exploration.ipynb`.

- Regex-based, lowercase, word-level tokenization; no stemming, lemmatization, or stopword removal.
- Negations and meaning-bearing words are preserved; unambiguous contractions and apostrophe variants are normalized.
- A fixed vocabulary is built from the **training split only**. Unavailable words become `[UNK]`.
- `[PAD]`, `[UNK]`, and `[SEP]` have IDs 0, 1, and 2.
- Examples are encoded without truncation and dynamically padded within each batch. Original lengths are retained for masking and GRU processing.

For the MLP and Standard GRU, the encoded input follows:

```text
O1 [SEP] O2 [SEP] H1 [SEP] H2
```

**Embeddings:** `fasttext-wiki-news-subwords-300` (loaded using Gensim), 300 dimensions, frozen during training.

| Vocabulary / embedding statistic | Value |
|---|---:|
| Vocabulary size, including special tokens | 15,038 |
| Training-derived vocabulary words | 15,035 |
| Training vocabulary coverage | 95.03% |
| Training token coverage | 99.62% |
| Embedding matrix shape | (15,038, 300) |

The `[PAD]` vector is zero; `[UNK]` uses the mean of available pretrained vectors; `[SEP]` uses a fixed random vector. Although the original fastText method supports character subwords, **this implementation performs fixed-vocabulary lookup**: unseen tokens map to `[UNK]` rather than receiving newly inferred subword vectors.

The preprocessing notebook verifies tokenizer consistency, vocabulary reconstruction, encoding, padding, lengths, and training-only vocabulary construction. The resulting artifact is saved as `project1/artifacts/fasttext_art.pt`.

## 3. Model architectures

### MLP — required architecture 1

The MLP independently mean-pools the embeddings of O1, O2, H1, and H2, concatenates the four 300-dimensional representations into a 1,200-dimensional vector, and applies a two-layer classifier:

```text
4 × mean-pooled fastText (300d) → concatenate (1200d)
→ Linear(1200, 128) → ReLU → Linear(128, 2)
```

The MLP has **153,986 trainable parameters**. It provides a simple baseline but does not preserve word order within a segment.

### Standard GRU — required architecture 2

The Standard GRU processes the encoded sequence using **two GRU layers** followed by a two-layer classifier. Unlike mean pooling, the recurrent encoder can represent word order and sequential context. It has **280,962 trainable parameters**.

Validation experiments also tested swapping H1 and H2. The Standard GRU showed **72.70% prediction-flip consistency**, indicating sensitivity to candidate order.

### PairwiseGRU — additional architecture

The PairwiseGRU constructs a candidate-specific representation for each hypothesis and applies the **same GRU encoder and scoring network** to both. The two candidate scores are compared to predict H1 or H2.

This shared-scoring design is symmetric with respect to hypothesis order. On validation data, prediction-flip consistency was **99.80%**; the remaining two examples produced tied scores (one due to `[UNK]` collisions and one with identical hypotheses). Excluding these ties, consistency was **100% (998/998)**. The PairwiseGRU has **280,833 trainable parameters**.

## 4. Training and validation experiments

All three original models use cross-entropy loss, the Adam optimizer, frozen fastText embeddings, and validation-based checkpoint selection. The principal seed-42 configurations are:

| Setting | MLP | Standard GRU | PairwiseGRU |
|---|---|---|---|
| Hidden dimension | 128 | 128 | 128 |
| GRU layers | — | 2 | 2 |
| Learning rate | 0.0003 | 0.001 | 0.001 |
| Dropout | 0.2 | 0.0 | 0.0 |
| Batch size | 32 | 32 | 32 |
| Optimizer | Adam | Adam | Adam |

The experiments include learning-rate, dropout, optimizer/weight-decay, and hidden-dimension comparisons, with early stopping and saved checkpoints. Some later configurations visibly overfit: training accuracy continues to rise while validation accuracy stalls or validation loss increases.

**Multi-seed validation results** (seeds 42, 123, and 2026):

| Architecture | Mean validation accuracy |
|---|---:|
| MLP, dropout 0.2 | 64.53% |
| Standard GRU, 128 hidden units | 68.03% |
| PairwiseGRU | **71.27%** |

PairwiseGRU exceeds the Standard GRU's mean validation accuracy by **3.23 percentage points**. This is a validation finding, not evidence of superior test performance.

## 5. Official test evaluation

Notebook `06_final_evaluation.ipynb` restores the original selected seed-42 checkpoints, reproduces their validation scores, and then evaluates each once on the **official test split**. Accuracy, balanced accuracy, macro-F1, cross-entropy loss, confusion matrices, and per-class recall are reported.

| Model | Test accuracy | Balanced accuracy | Macro-F1 | Test loss |
|---|---:|---:|---:|---:|
| MLP | 51.57% | 51.54% | 0.5154 | 1.0770 |
| Standard GRU | **54.05%** | **54.09%** | **0.5404** | **0.9039** |
| PairwiseGRU | 53.98% | 53.98% | 0.5398 | 1.2057 |

The Standard GRU narrowly outperforms PairwiseGRU on test accuracy (by **0.07 percentage points**), despite the latter's higher validation score. The original expected accuracies documented in `EXPECTATIONS.md` (68% for the MLP and 75% for the RNN) were predictions, **not measurements**, and were not achieved.

## 6. Generalization and error analysis

### Dataset shift

Label balance, average sequence length, and unknown-token rates are similar between validation and test. In particular, both validation and test have an unknown-token rate of **1.63%**, and the audit detected no exact observation-pair overlap across splits.

However, the **candidate hypotheses** differ substantially:

| Hypothesis statistic | Train | Validation | Test |
|---|---:|---:|---:|
| Mean Jaccard token similarity | 0.602 | 0.617 | **0.225** |
| Median Jaccard similarity | 0.667 | 0.667 | **0.167** |
| Mean absolute length difference (tokens) | 1.19 | 1.08 | **3.10** |

This suggests a meaningful shift in the kinds of candidate explanations presented at test time. It is a plausible contributor to the validation-to-test performance gap, but does not by itself establish causality.

### Controlled development holdouts

For robustness experiments, the original training data was divided into **15,121 development-training examples**, an **840-example normal holdout**, and an **840-example low-similarity hard holdout**. A separate **14,440-token vocabulary** was constructed using only development-training data. New models were trained without access to either development holdout.

The experiments explored batch sizes, LayerNorm, similarity-weighted sampling, dropout, and hypothesis length-balanced loss weighting. Notable observations:

- **Batch size:** Size 64 yielded the highest normal-holdout accuracy in the PairwiseGRU batch-size comparison (72.26%), but not the best hard-holdout accuracy for that checkpoint.
- **LayerNorm:** At batch size 64, the accuracy-selected LayerNorm model reached 69.40% on the hard holdout versus 63.45% without LayerNorm, while normal-holdout accuracy decreased.
- **Similarity weighting:** Giving low-similarity training examples 3× sampling weight improved the hard-holdout results in the development comparison.
- **Length balancing:** Over three seeds, length-balanced loss weighting increased mean hard-holdout accuracy by **0.87 pp for MLP** and **1.83 pp for Standard GRU**, while reducing length-related accuracy differences.

The artificially constructed hard holdout is a useful diagnostic, but it is **not a substitute for the official test distribution**.

### Exploratory test comparisons

Further tests in Notebook 07 were conducted **after** the original official test evaluation, and are therefore descriptive/exploratory:

- A similarity-weighted PairwiseGRU checkpoint reached **54.31% test accuracy**, only **0.26 pp** above the original Standard GRU.
- Across three development-trained seeds, length balancing changed mean MLP test accuracy from **51.94% to 52.59%**, but mean Standard GRU test accuracy from **53.05% to 52.70%**.
- Increasing dropout did not consistently improve generalization; one GRU run collapsed to predicting the same class for every example.

These results **do not demonstrate a reliable solution** to the generalization gap. Models trained on different data subsets and vocabularies are not strictly like-for-like comparisons with the original checkpoints.

### Error analysis

An exploratory six-checkpoint analysis uses three length-balanced MLP models and three uniformly trained GRU models (seeds 42, 43, 44). Across the 1,532 official test examples, **404 examples (26.37%) were incorrectly predicted by all six individual checkpoints**. Majority-vote predictions from both architectures were wrong on 519 examples (33.88%).

Notebook 07 exports prediction-level CSVs, shared and architecture-specific error examples, and files for manual review. The analysis suggests that hypothesis length and lexical overlap are related to performance, but identifying specific reasoning failures requires qualitative examination.

## 7. Repository structure

```text
project1/
├── src/
│   ├── preprocessing.py        # Tokenization and vocabulary
│   ├── data_loader.py          # Encoding and dynamic padding
│   ├── models.py               # MLP, Standard GRU, PairwiseGRU
│   └── training.py             # Training and evaluation helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embeddings_mlp.ipynb
│   ├── 03_gru.ipynb
│   ├── 04_gru_tuning.ipynb
│   ├── 05_pairwise_gru.ipynb
│   ├── 06_final_evaluation.ipynb
│   └── 07_generalization_experiments.ipynb
└── artifacts/                  # Generated embeddings, checkpoints, CSVs
```

The tree describes the **paths used by the notebooks**. Check the repository for any additional submission files or a different directory layout.

## 8. Running the project

Use a Python environment with **PyTorch, Hugging Face Datasets, Gensim, NumPy, pandas, scikit-learn, Matplotlib, and W&B**; Jupyter is needed to run the notebooks. CUDA-capable NVIDIA hardware is required by some training notebooks as currently written. The first notebook also downloads a large pretrained fastText model.

From the repository root, install the required libraries if they are not already available:

```bash
python -m pip install torch datasets gensim numpy pandas scikit-learn matplotlib wandb jupyterlab
```

Launch Jupyter and follow the notebooks in numerical order:

```bash
jupyter lab
```

1. **01 — Data exploration:** Inspect data, construct and verify the training-only vocabulary, save `fasttext_art.pt`.
2. **02 — Embeddings + MLP:** Train and tune the MLP baseline.
3. **03 — GRU:** Implement and verify the initial GRU baseline.
4. **04 — GRU tuning:** Tune the GRU; investigate hypothesis-order sensitivity and augmentation.
5. **05 — PairwiseGRU:** Test swap invariance, train across seeds, and investigate robustness.
6. **06 — Final evaluation:** Restore the original checkpoints and report their official test results.
7. **07 — Generalization experiments:** Analyze dataset shift, robustness, length bias, alternative training strategies, and errors.

**Reproduction notes:** Several training experiments are computationally expensive and their outputs are saved in the notebooks. Their source cells may be annotated to discourage accidental reruns. Checkpoint-loading cells require the corresponding `.pt` files under `project1/artifacts/`. These generated files are not necessarily stored in Git; use the saved artifacts or rerun the relevant training cells if necessary. W&B tracking requires an authenticated account for new runs.

## 9. Tools and AI Assistance

This project was implemented in Python using PyTorch for model development and training, pretrained fastText embeddings for word representations, Hugging Face for dataset access, and Weights & Biases for experiment tracking.

AI Assistance:
- ChatGPT (OpenAI): Used to explain NLP concepts, support code understanding and debugging, discuss model architectures and experimental results, and assist with documentation and presentation preparation.

## 10. Experiment tracking

Training configurations, metrics, checkpoint references, and experiment comparisons are logged in the project's W&B workspace:

**[NLP Project 1 — Weights & Biases](https://forge.coreweave.com/wandb/kimbortis-hochschule-luzern/nlp-art-hs26-project1)**

Notebook outputs preserve key numerical results independently of the W&B interface.

## 11. Conclusions and limitations

The project implements both required architectures and an additional hypothesis-order-symmetric PairwiseGRU. While the PairwiseGRU improves validation accuracy and resolves most answer-order inconsistency, the Standard GRU performs best in the original official test evaluation.

The major limitation is **generalization across dataset distributions**. Test hypotheses differ substantially from training and validation in lexical overlap and relative length. Controlled experiments with LayerNorm, sampling strategies, dropout, and length-balanced losses yield mixed improvements but no convincing resolution of the gap.

Additional limitations include fixed-vocabulary `[UNK]` handling, overfitting in some training runs, limited seeds for several robustness experiments, and repeated exploratory use of the official test set after the initial final evaluation. Accordingly, exploratory test improvements are **not claimed as independently validated advances**.

The central takeaway is that **higher validation accuracy and better architectural symmetry do not necessarily imply better out-of-distribution test performance**.
