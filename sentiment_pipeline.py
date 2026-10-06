"""
Apple-tweet sentiment classification: baselines -> tuned linear model -> BiLSTM.

Task      : binary sentiment (negative=0 / positive=1) on the Crowdflower
            "Apple Twitter Sentiment" data (neutral + not_relevant rows dropped).
Design    : * dedupe BEFORE splitting (retweets would otherwise leak across splits)
            * stratified train/test split, test set touched exactly once
            * class weights instead of undersampling (keeps all the data)
            * models are compared on the SAME held-out test set with
              imbalance-aware metrics (accuracy alone is misleading here: the
              majority-class baseline already scores ~74%)
            * bootstrap 95% CIs, since the test set is small
            * ablation: same model on raw vs. cleaned text, to measure what the
              preprocessing actually buys

Usage     : python sentiment_pipeline.py --data "Sentiment Analysis Dataset.csv"
            python sentiment_pipeline.py --data "..." --skip-nn      # no TensorFlow
"""
import argparse
import html
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Data loading + text cleaning
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"\brt\b")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # "soooooo" -> "soo"
NON_TEXT_RE = re.compile(r"[^a-z0-9!?' ]+")


def clean_text(text: str) -> str:
    """Tweet-specific normalisation. Keeps '!' and '?' (strong sentiment cues)."""
    t = html.unescape(str(text)).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)  # @apple etc. carries no sentiment
    t = RT_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r" \1 ", t)  # keep the hashtag word
    t = REPEAT_RE.sub(r"\1\1", t)
    t = NON_TEXT_RE.sub(" ", t)
    t = re.sub(r"([!?])", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_data(path: str):
    df = pd.read_csv(path, encoding="latin1")[["sentiment", "text"]].copy()
    stats = {"raw_rows": len(df)}

    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["sentiment"].isin(["1", "5"])].copy()  # drop neutral (3) + not_relevant
    df["label"] = (df["sentiment"] == "5").astype(int)
    stats["pos_neg_rows"] = len(df)

    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0]

    # Same cleaned text with conflicting labels -> ambiguous, drop all copies.
    n_labels = df.groupby("clean")["label"].nunique()
    df = df[~df["clean"].isin(n_labels[n_labels > 1].index)]
    # Exact/retweet duplicates -> keep one (prevents train/test leakage).
    df = df.drop_duplicates("clean").reset_index(drop=True)

    stats["final_rows"] = len(df)
    stats["class_counts"] = {
        "negative": int((df["label"] == 0).sum()),
        "positive": int((df["label"] == 1).sum()),
    }
    return df, stats


# --------------------------------------------------------------------------- #
# 2. Metrics
# --------------------------------------------------------------------------- #
def evaluate(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "f1_neg": f1_score(y, pred, pos_label=0),
        "f1_pos": f1_score(y, pred, pos_label=1),
        "roc_auc": roc_auc_score(y, prob),
    }


def bootstrap_ci(y, prob, metric, n=1000, thr=0.5, seed=SEED):
    """95% percentile bootstrap CI for a label-based metric."""
    rng = np.random.default_rng(seed)
    pred = (prob >= thr).astype(int)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #
def tuned_logreg() -> GridSearchCV:
    """TF-IDF (word + char n-grams) -> class-weighted logistic regression."""
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("tfidf", features),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    grid = {
        "tfidf__word__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.3, 1, 3, 10],
    }
    return GridSearchCV(
        pipe,
        grid,
        scoring="f1_macro",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )


def run_bilstm(X_train, y_train, X_test, seed, max_epochs=40):
    """Bidirectional LSTM with in-graph TextVectorization (fit on train only)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(seed)
    X_train, X_test = np.array(X_train, dtype=str), np.array(X_test, dtype=str)
    y_train = np.asarray(y_train)

    # Hold out a stratified validation slice for early stopping.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
    )

    vec = layers.TextVectorization(
        max_tokens=5000, output_sequence_length=40, standardize=None
    )
    vec.adapt(X_fit)  # vocabulary from the fitting split only -> no leakage

    model = keras.Sequential(
        [
            keras.Input(shape=(), dtype="string"),
            vec,
            layers.Embedding(vec.vocabulary_size(), 64, mask_zero=True),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.5),
            layers.Dense(
                32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)
            ),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_fit)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    def ds(X, y=None, shuffle=False):
        d = tf.data.Dataset.from_tensor_slices(X if y is None else (X, y))
        if shuffle:
            d = d.shuffle(len(X), seed=seed)
        return d.batch(32)

    hist = model.fit(
        ds(X_fit, y_fit, shuffle=True),
        validation_data=ds(X_val, y_val),
        epochs=max_epochs,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    prob = model.predict(ds(X_test), verbose=0).ravel()
    return prob, hist.history


# --------------------------------------------------------------------------- #
# 4. Reporting helpers
# --------------------------------------------------------------------------- #
def save_confusion_matrices(y_test, probs: dict, out: Path):
    fig, axes = plt.subplots(1, len(probs), figsize=(4.2 * len(probs), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, probs.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            (p >= 0.5).astype(int),
            display_labels=["neg", "pos"],
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def save_training_curve(history: dict, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(history["accuracy"], label="train")
    ax[1].plot(history["val_accuracy"], label="val")
    ax[1].set(title="Accuracy", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "bilstm_training_curve.png", dpi=150)
    plt.close(fig)


def top_features(grid: GridSearchCV, k=15):
    pipe = grid.best_estimator_
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    word_idx = [i for i, n in enumerate(names) if n.startswith("word__")]
    order = sorted(word_idx, key=lambda i: coef[i])
    neg = [names[i].replace("word__", "") for i in order[:k]]
    pos = [names[i].replace("word__", "") for i in order[::-1][:k]]
    return neg, pos


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Sentiment Analysis Dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3, help="BiLSTM seeds (ensembled)")
    ap.add_argument("--skip-nn", action="store_true", help="skip the TensorFlow model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    # ---- data ----
    df, stats = load_data(args.data)
    print("DATA:", json.dumps(stats, indent=2))

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    y_train, y_test = train_df["label"].values, test_df["label"].values
    print(f"train={len(train_df)}  test={len(test_df)}")

    probs, rows = {}, []

    # ---- baseline 0: majority class ----
    dummy = DummyClassifier(strategy="prior").fit(train_df["clean"], y_train)
    probs["Majority class"] = dummy.predict_proba(test_df["clean"])[:, 1]

    # ---- ablation: tuned LR on raw text vs cleaned text ----
    lr_raw = tuned_logreg().fit(train_df["text"], y_train)
    probs["TF-IDF+LR (raw text)"] = lr_raw.predict_proba(test_df["text"])[:, 1]

    lr_clean = tuned_logreg().fit(train_df["clean"], y_train)
    probs["TF-IDF+LR (cleaned)"] = lr_clean.predict_proba(test_df["clean"])[:, 1]
    print("LR best params:", lr_clean.best_params_, f"| CV macro-F1={lr_clean.best_score_:.3f}")

    # ---- BiLSTM (multi-seed ensemble) ----
    seed_scores, history = [], None
    if not args.skip_nn:
        seed_probs = []
        for s in range(args.seeds):
            p, h = run_bilstm(train_df["clean"], y_train, test_df["clean"], SEED + s)
            seed_probs.append(p)
            seed_scores.append(evaluate(y_test, p)["macro_f1"])
            history = history or h
        probs["BiLSTM (ensemble)"] = np.mean(seed_probs, axis=0)
        print(
            f"BiLSTM per-seed macro-F1: {np.round(seed_scores, 3)} "
            f"(mean {np.mean(seed_scores):.3f} +/- {np.std(seed_scores):.3f})"
        )

    # ---- evaluate everything on the same held-out test set ----
    for name, p in probs.items():
        m = evaluate(y_test, p)
        lo, hi = bootstrap_ci(
            y_test, p, lambda a, b: f1_score(a, b, average="macro")
        )
        rows.append({"model": name, **m, "macro_f1_95CI": f"[{lo:.3f}, {hi:.3f}]"})

    results = pd.DataFrame(rows).set_index("model").round(3)
    print("\nTEST RESULTS\n", results.to_string())
    results.to_csv(out / "metrics.csv")

    # ---- artefacts ----
    save_confusion_matrices(
        y_test, {k: v for k, v in probs.items() if k != "Majority class"}, out
    )
    if history:
        save_training_curve(history, out)

    neg_words, pos_words = top_features(lr_clean)
    print("\nMost negative words:", neg_words)
    print("Most positive words:", pos_words)

    # error analysis: most confidently wrong predictions of the best linear model
    best_p = probs["TF-IDF+LR (cleaned)"]
    err = test_df.assign(prob_pos=best_p, pred=(best_p >= 0.5).astype(int))
    err = err[err["pred"] != err["label"]].copy()
    err["confidence"] = np.where(err["pred"] == 1, err["prob_pos"], 1 - err["prob_pos"])
    err.sort_values("confidence", ascending=False)[
        ["text", "label", "pred", "confidence"]
    ].head(25).to_csv(out / "top_errors.csv", index=False)

    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "data": stats,
                "lr_best_params": {k: str(v) for k, v in lr_clean.best_params_.items()},
                "lr_cv_macro_f1": lr_clean.best_score_,
                "bilstm_seed_macro_f1": seed_scores,
                "top_negative_words": neg_words,
                "top_positive_words": pos_words,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics, plots and error analysis to ./{out}/")


if __name__ == "__main__":
    main()
