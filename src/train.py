"""
Train and evaluate machine learning models for fake news detection.

Trains three classical ML models on TF-IDF features (Logistic Regression,
Passive Aggressive Classifier, Random Forest), compares them, saves the
best-performing model + vectorizer, and produces evaluation plots.

Usage:
    python src/train.py
"""
import os
import sys

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")  # safe for headless environments
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess_dataframe

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """
    Loads the full Kaggle "Fake and Real News" dataset if present
    (data/Fake.csv and data/True.csv), otherwise falls back to the
    small bundled sample dataset (data/sample_data.csv) so you can
    test the pipeline immediately.
    """
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    sample_path = os.path.join(DATA_DIR, "sample_data.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("Loading full Kaggle dataset (Fake.csv + True.csv)...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        fake_df["label"] = "FAKE"
        true_df["label"] = "REAL"
        df = pd.concat([fake_df, true_df], ignore_index=True)
    elif os.path.exists(sample_path):
        print("NOTE: Kaggle CSVs not found in data/ -- using the bundled "
              "sample_data.csv instead. This is a small synthetic dataset "
              "meant only for testing that the pipeline runs correctly. "
              "See README.md to download the full dataset for real results.")
        df = pd.read_csv(sample_path)
    else:
        raise FileNotFoundError(
            "No dataset found. Either place Fake.csv and True.csv in the "
            "data/ folder (see README.md for the download link), or run "
            "`python data/generate_sample_data.py` to create a test dataset."
        )

    df = df.dropna(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def train_and_evaluate():
    df = load_data()
    title_col = "title" if "title" in df.columns else None
    df = preprocess_dataframe(df, text_column="text", title_column=title_col)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    best_model_name, best_model, best_acc = None, None, -1.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"{name} accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        plot_confusion_matrix(y_test, preds, name)

        if acc > best_acc:
            best_acc, best_model, best_model_name = acc, model, name

    print(f"\nBest model: {best_model_name} ({best_acc:.4f} accuracy)")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    with open(os.path.join(MODEL_DIR, "model_info.txt"), "w") as f:
        f.write(f"Best model: {best_model_name}\nAccuracy: {best_acc:.4f}\n\n")
        f.write("All results:\n")
        for name, acc in results.items():
            f.write(f"  {name}: {acc:.4f}\n")

    print(f"\nSaved best model ('{best_model_name}') and vectorizer to {MODEL_DIR}/")
    plot_model_comparison(results)

    return results


def plot_confusion_matrix(y_test, preds, model_name):
    cm = confusion_matrix(y_test, preds, labels=["FAKE", "REAL"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.title(f"Confusion Matrix -- {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{fname}.png"))
    plt.close()


def plot_model_comparison(results):
    plt.figure(figsize=(6, 4))
    names = list(results.keys())
    accs = list(results.values())
    plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Accuracy")
    plt.title("Model Comparison -- Fake News Detection")
    plt.ylim(0, 1)
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.02, f"{acc:.3f}", ha="center")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()