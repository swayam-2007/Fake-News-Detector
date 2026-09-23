"""
Load the trained model and classify a single piece of news text
from the command line.

Usage:
    python src/predict.py "Some news headline or article text here..."

    Or run with no arguments for interactive mode:
    python src/predict.py
"""
import os
import sys

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        confidence = proba[classes.index(pred)]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        confidence = 1 / (1 + np.exp(-abs(score)))  # rough confidence squashing

    return pred, confidence


def _print_result(text, model, vectorizer):
    label, conf = predict(text, model, vectorizer)
    conf_str = f" (confidence: {conf:.2%})" if conf is not None else ""
    print(f"Prediction: {label}{conf_str}")


if __name__ == "__main__":
    loaded_model, loaded_vectorizer = load_artifacts()

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        _print_result(input_text, loaded_model, loaded_vectorizer)
    else:
        print("Enter news text to classify (Ctrl+C to exit):\n")
        while True:
            try:
                input_text = input("> ")
                if not input_text.strip():
                    continue
                _print_result(input_text, loaded_model, loaded_vectorizer)
                print()
            except KeyboardInterrupt:
                print("\nExiting.")
                break