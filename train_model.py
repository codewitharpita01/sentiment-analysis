"""Train, evaluate, and save the local TF-IDF + Logistic Regression model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_predictions
from src.prediction import LABELS
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    required = {"text", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain columns named 'text' and 'sentiment'.")
    data = data[["text", "sentiment"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data["sentiment"] = data["sentiment"].astype(str).str.strip().str.title()
    data = data[(data.text != "") & data.sentiment.isin(LABELS)].drop_duplicates()
    if data.empty:
        raise ValueError("No valid labeled rows found. Labels must be Positive, Negative, or Neutral.")
    if data.sentiment.nunique() < 2:
        raise ValueError("Training requires at least two different sentiment classes.")
    if len(data) < 3:
        raise ValueError("Training and evaluation need at least three valid rows across two or more classes.")
    return data.reset_index(drop=True)


def train(data_path: Path, model_path: Path) -> dict:
    data = load_dataset(data_path)
    x = data.text
    y = data.sentiment
    counts = y.value_counts()
    # Stratification needs at least two examples per class and enough test slots.
    test_rows = max(y.nunique(), round(len(data) * 0.2))
    can_stratify = counts.min() >= 2 and len(data) - test_rows >= y.nunique()
    stratify = y if can_stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_rows, random_state=42, stratify=stratify
    )
    if y_train.nunique() < 2:
        raise ValueError(
            "The training split contains only one class. Add more examples for each class "
            "or use a larger dataset so the model can learn at least two classes."
        )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=1,
                                  sublinear_tf=True, max_features=50000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(x_train, y_train)
    metrics = evaluate_predictions(y_test, pipeline.predict(x_test))
    artifact = {
        "pipeline": pipeline,
        "labels": list(LABELS),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "test_metrics": metrics,
        "data_source": str(data_path.name),
        "stratified_split": bool(can_stratify),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sample_reviews.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "sentiment_pipeline.joblib")
    args = parser.parse_args()
    artifact = train(args.data, args.model)
    metrics = artifact["test_metrics"]
    print(f"Saved pipeline: {args.model}")
    print(f"Held-out accuracy ({artifact['test_rows']} rows): {metrics['accuracy']:.3f}")
    print(f"Stratified split: {artifact['stratified_split']}")
    print(pd.DataFrame(metrics["report"]).T.round(3).to_string())
    print("Confusion matrix (actual rows, predicted columns; Positive, Negative, Neutral):")
    print(pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
