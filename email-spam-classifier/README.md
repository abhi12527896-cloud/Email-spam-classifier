# Email/SMS Spam Classifier

A text classification project that flags spam messages using TF-IDF features and a
Multinomial Naive Bayes classifier. Built with Python, scikit-learn, Pandas, and Matplotlib.

## How it works

1. **Data cleaning** — duplicates removed, text lowercased, URLs/punctuation/numbers stripped.
2. **Feature extraction** — raw text converted into TF-IDF vectors (with English stop words removed).
3. **Model** — a Multinomial Naive Bayes classifier trained on the vectorized text.
4. **Evaluation** — accuracy, precision/recall, and a confusion matrix plot.
5. **Persistence** — the trained model + vectorizer are saved together with `joblib` so they
   can be reloaded instantly without retraining.

## Project structure

```
email-spam-classifier/
├── data/
│   └── spam_sample.csv     # small demo dataset (see note below)
├── preprocess.py            # text cleaning helper
├── train.py                 # trains the model, prints metrics, saves model + confusion matrix
├── predict.py                # classifies a single new message using the saved model
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model:
```bash
python train.py --data data/spam_sample.csv
```

This prints accuracy and a classification report, saves the confusion matrix to
`outputs/confusion_matrix.png`, and saves the trained model to `models/spam_model.joblib`.

Classify a new message:
```bash
python predict.py --model models/spam_model.joblib --message "You have won a free prize, click here!"
```

## About the dataset

`data/spam_sample.csv` is a small (~70 message) demo set included so the pipeline runs
out of the box. For results that match a production-grade model (high-90s% accuracy on
5,000+ messages), download the full **UCI SMS Spam Collection** dataset (also mirrored on
Kaggle as "SMS Spam Collection Dataset"), save it as `data/spam.csv` with `label,message`
columns, and run:

```bash
python train.py --data data/spam.csv
```

## Possible extensions

- Swap `MultinomialNB` for `LogisticRegression` or a `LinearSVC` and compare performance.
- Add n-gram features (`ngram_range=(1, 2)`) to the `TfidfVectorizer`.
- Wrap `predict.py` in a small Flask/FastAPI app for a live demo.
