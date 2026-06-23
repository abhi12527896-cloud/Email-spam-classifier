"""
Text preprocessing utilities for the spam classifier.
"""
import re
import string


def clean_text(text: str) -> str:
    """
    Normalize a raw message for vectorization:
    - lowercase
    - strip URLs
    - remove punctuation / special characters
    - collapse extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
