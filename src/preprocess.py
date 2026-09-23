"""
Text preprocessing utilities for the Fake News Detector project.

No external downloads required (uses scikit-learn's built-in stopword
list instead of NLTK, so this works offline out of the box).
"""
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """
    Clean a raw piece of text:
      - lowercase
      - remove URLs and HTML tags
      - remove punctuation and digits
      - remove stopwords and very short tokens
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)                     # URLs
    text = re.sub(r"<.*?>", " ", text)                                      # HTML tags
    text = re.sub(r"\d+", " ", text)                                        # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))        # punctuation
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_dataframe(df, text_column="text", title_column="title",
                          output_column="clean_text"):
    """
    Combine title + text (if a title column exists) and clean it,
    adding a new column with the result.
    """
    df = df.copy()
    if title_column and title_column in df.columns:
        df["combined_text"] = df[title_column].fillna("") + " " + df[text_column].fillna("")
    else:
        df["combined_text"] = df[text_column].fillna("")

    df[output_column] = df["combined_text"].apply(clean_text)
    return df