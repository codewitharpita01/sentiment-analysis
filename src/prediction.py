"""Loading a trained model and producing review predictions."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

LABELS = ("Positive", "Negative", "Neutral")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment_pipeline.joblib"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}. Run: python train_model.py")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Model file has an unsupported format. Retrain it with train_model.py.")
    return artifact


def predict_reviews(reviews: list[str], artifact: dict[str, Any]) -> pd.DataFrame:
    """Return labels and classifier probabilities for non-empty review strings."""
    cleaned = [str(review).strip() for review in reviews]
    if any(not review for review in cleaned):
        raise ValueError("Reviews cannot be empty.")
    pipeline = artifact["pipeline"]
    labels = pipeline.predict(cleaned)
    result = pd.DataFrame({"Review": cleaned, "Predicted sentiment": labels})
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(cleaned)
        result["Confidence"] = probabilities.max(axis=1)
        result["Confidence"] = result["Confidence"].map(lambda score: f"{score:.1%}")
    else:
        result["Confidence"] = "Unavailable"
    result.insert(0, "Row", range(1, len(result) + 1))
    return result
