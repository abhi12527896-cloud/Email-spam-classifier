"""
Classify new messages with a trained spam model.

Usage:
    python predict.py --model models/spam_model.joblib --message "You have won a free prize!"
    python predict.py --model models/spam_model.joblib --message "See you at 6pm"
"""
import argparse
import os

import joblib

from preprocess import clean_text


def main():
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("--model", default="models/spam_model.joblib", help="Path to the trained model file.")
    parser.add_argument("--message", required=True, help="The message text to classify.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model file not found at '{args.model}'. Train one first with: python train.py"
        )

    bundle = joblib.load(args.model)
    model, vectorizer = bundle["model"], bundle["vectorizer"]

    cleaned = clean_text(args.message)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)

    print(f"Message:    {args.message}")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
