"""
Train a Multinomial Naive Bayes spam classifier on an SMS/email dataset.

Usage:
    python train.py --data data/spam_sample.csv
    python train.py --data data/spam.csv --model-out models/spam_model.joblib
"""
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV with 'label' and 'message' columns and clean it up."""
    df = pd.read_csv(path, encoding="latin-1")

    # Support the raw UCI SMS Spam Collection format (v1/v2 columns, no header)
    if "label" not in df.columns or "message" not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates(subset=["message"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train the spam classifier.")
    parser.add_argument("--data", default="data/spam_sample.csv", help="Path to labeled CSV (label, message).")
    parser.add_argument("--model-out", default="models/spam_model.joblib", help="Where to save the trained pipeline.")
    parser.add_argument("--cm-out", default="outputs/confusion_matrix.png", help="Where to save the confusion matrix plot.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} messages after cleaning/deduplication "
          f"({(df['label'] == 'spam').sum()} spam / {(df['label'] == 'ham').sum()} ham).")

    df["clean_message"] = df["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label"],
        test_size=args.test_size, random_state=args.random_state, stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.cm_out), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Spam Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.cm_out, dpi=150)
    print(f"Confusion matrix saved to {args.cm_out}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, args.model_out)
    print(f"Trained model + vectorizer saved to {args.model_out}")


if __name__ == "__main__":
    main()
