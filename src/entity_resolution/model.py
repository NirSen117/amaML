"""Trainable pair classifier and portable persistence.

CatBoost is preferred when installed (GPU first by default); failures to load
CatBoost or initialize its GPU backend fall back to CPU CatBoost and then to
scikit-learn logistic regression.
"""
import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from .features import FEATURE_NAMES, feature_matrix

LOGGER = logging.getLogger(__name__)

class PairModel:
    def __init__(self, classifier=None, model_type="logistic", device="cpu"):
        self.classifier = classifier
        self.model_type = model_type
        self.device = device

    @staticmethod
    def _logistic():
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    @staticmethod
    def _catboost(device, random_state):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=7, learning_rate=0.08,
            loss_function="Logloss", random_seed=random_state,
            task_type="GPU" if device == "gpu" else "CPU",
            verbose=False,
        )

    def fit(self, pairs, labels, random_state=42):
        LOGGER.info("starting model fitting for %d pairs", len(pairs))
        features = feature_matrix(pairs)
        labels_array = np.asarray(labels)
        if self.model_type == "logistic":
            self.classifier = self._logistic()
            self.classifier.fit(features, labels_array)
            self.device = "cpu"
            return self

        try:
            import catboost  # noqa: F401 - optional dependency
            devices = ["cpu"] if self.device == "cpu" else ["gpu", "cpu"]
            for device in devices:
                try:
                    self.classifier = self._catboost(device, random_state)
                    LOGGER.info("fitting CatBoost on %s", device.upper())
                    self.classifier.fit(features, labels_array)
                    self.model_type, self.device = "catboost", device
                    LOGGER.info("using CatBoost on %s", device.upper())
                    return self
                except Exception as exc:
                    LOGGER.warning("CatBoost %s training unavailable: %s", device.upper(), exc)
        except Exception as exc:
            LOGGER.warning("CatBoost import unavailable: %s", exc)

        self.classifier = self._logistic()
        self.classifier.fit(features, labels_array)
        self.model_type, self.device = "logistic", "cpu"
        LOGGER.info("using scikit-learn LogisticRegression fallback")
        return self

    def predict_proba(self, pairs):
        return self.classifier.predict_proba(feature_matrix(pairs))[:, 1]

    def save(self, path: Path):
        LOGGER.info("saving model to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": self.classifier, "features": FEATURE_NAMES,
                     "model_type": self.model_type, "device": self.device}, path)
        LOGGER.info("model saved to %s", path)

    @classmethod
    def load(cls, path: Path):
        LOGGER.info("loading model from %s", path)
        payload = joblib.load(path)
        if tuple(payload["features"]) != tuple(FEATURE_NAMES):
            raise ValueError("Saved model feature schema does not match this package")
        return cls(payload["classifier"], payload.get("model_type", "logistic"),
                   payload.get("device", "cpu"))
