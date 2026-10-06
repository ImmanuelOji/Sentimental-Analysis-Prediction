# Tweet Sentiment Classification: Baselines vs. BiLSTM

Binary sentiment classification (negative / positive) on Apple-related tweets. The project compares a majority-class baseline, a tuned TF-IDF + logistic regression model, and a BiLSTM on the same held-out test set, using imbalance-aware metrics and bootstrap confidence intervals.

**Tech:** Python, scikit-learn, TensorFlow/Keras, pandas, matplotlib

## Key results

Held-out test set: 278 tweets (stratified 80/20 split, seed 42).

| Model | Accuracy | Balanced acc. | Macro-F1 | Macro-F1 95% CI | ROC-AUC |
|---|---|---|---|---|---|
| Majority class | 0.745 | 0.500 | 0.427 | [0.410, 0.442] | 0.500 |
| TF-IDF + LR (raw text) | 0.856 | 0.797 | 0.805 | [0.751, 0.856] | 0.920 |
| **TF-IDF + LR (cleaned text)** | **0.856** | **0.815** | **0.813** | [0.758, 0.861] | 0.912 |
| BiLSTM (3-seed ensemble) | 0.817 | 0.798 | 0.775 | [0.722, 0.823] | 0.882 |

Macro-F1 rose from 0.43 (majority-class baseline) to 0.81 with the tuned linear model. The 5-fold cross-validated macro-F1 on the training data was 0.754, so the true performance is probably in the 0.75-0.82 range. The test set is small, and the confidence intervals reflect that.

![Confusion matrices](results/confusion_matrices.png)

## Why these metrics

The dataset is imbalanced (74% negative), so a model that always predicts "negative" already scores 74.5% accuracy. Accuracy alone would make weak models look good, so the comparison uses macro-F1 and balanced accuracy, with bootstrap confidence intervals on the test set.

## Data

- Apple-related tweets with crowd-annotated sentiment labels, loaded from `Sentiment Analysis Dataset.csv` (not included in this repo; place it in the repo root to reproduce).
- 3,886 raw tweets, of which 1,642 are labeled positive or negative. Neutral and not-relevant rows are dropped.
- After removing empty, ambiguous (same text, conflicting labels) and duplicate or retweeted tweets: **1,386 tweets** (1,032 negative, 354 positive).

## Method

1. **Deduplication before splitting.** Duplicate tweets and retweets are removed so the same text cannot appear in both train and test.
2. **Stratified split.** 80% train / 20% test. The test set is evaluated once.
3. **Text cleaning.** Lowercasing, HTML unescaping, removal of URLs, @mentions and "RT", hashtag words kept, repeated letters collapsed ("soooo" to "soo"), and "!" and "?" kept as sentiment cues.
4. **TF-IDF + logistic regression.** Word and character n-gram features, class-weighted loss, and a grid search over the n-gram range and regularization strength using 5-fold stratified cross-validation (macro-F1).
5. **BiLSTM.** In-graph `TextVectorization` fitted on training data only, embedding layer, bidirectional LSTM, dropout, class weights, early stopping on a held-out validation slice, and 3 random seeds averaged into an ensemble.
6. **Evaluation.** All models are scored on the same test set. A preprocessing ablation runs the same linear model on raw and cleaned text. Bootstrap resampling (1,000 resamples) gives 95% confidence intervals for macro-F1.

## Findings

- **The simple model won.** Tuned TF-IDF + logistic regression (macro-F1 0.813) beat the BiLSTM ensemble (0.775). With roughly 1,100 training tweets, sparse n-gram features are hard to beat with a recurrent network. Individual BiLSTM seeds ranged from 0.740 to 0.788.
- **Cleaning had a small, statistically unclear effect.** Raw text scored 0.805 macro-F1 and cleaned text scored 0.813, well inside the overlapping confidence intervals. The gains over the baseline come mostly from class weighting, n-gram features and tuning, not from the cleaning step.
- **Learned features are sensible.** Most negative: *fuck, shit, stop, hate, wtf*. Most positive: *love, best, great, awesome, amazing, thanks*. A few high-weight function words (*for, be, tv*) are typical small-data noise.
- **The positive class is the hard one.** F1 is 0.90 for negative tweets and 0.72 for positive tweets.

## Error analysis

The 25 most confidently wrong predictions of the linear model are saved to `results/top_errors.csv` for inspection.

## Limitations

- Small dataset (1,386 tweets, 278 in the test set), so metrics have wide confidence intervals and could shift with a different split.
- Binary task only. Neutral tweets were removed, so the model cannot predict "neither".
- Labels come from crowd annotators and may be noisy.
- Sarcasm and negation are not handled explicitly.

## Possible improvements

- Pretrained embeddings (e.g. GloVe) or a fine-tuned transformer for the neural model.
- Repeated stratified cross-validation for a tighter estimate than a single split.
- Probability calibration and threshold tuning for the positive class.
- A three-class version that keeps the neutral tweets.

## Reproduce

Download or clone this repository, then from its root folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place "Sentiment Analysis Dataset.csv" in the repo root, then:
python sentiment_pipeline.py --data "Sentiment Analysis Dataset.csv"

# without TensorFlow (skips the BiLSTM):
python sentiment_pipeline.py --data "Sentiment Analysis Dataset.csv" --skip-nn
```

Outputs are written to `results/`: `metrics.csv`, `confusion_matrices.png`, `bilstm_training_curve.png`, `top_errors.csv`, `summary.json`. A fixed random seed (42) is used throughout; BiLSTM results can vary slightly across machines and TensorFlow versions.

## Repository structure

```
.
├── sentiment_pipeline.py     # full pipeline: data, models, evaluation, reporting
├── requirements.txt
├── README.md
└── results/                  # generated outputs (metrics, plots, error analysis)
```
