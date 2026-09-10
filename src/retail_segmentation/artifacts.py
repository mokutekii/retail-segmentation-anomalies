"""Portable model artifacts for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SegmentationBundle:
    """A fitted transformation, segmentation model, anomaly model, and feature contract."""

    scaler: Any
    kmeans: Any
    anomaly_model: Any
    feature_columns: tuple[str, ...]

    def _prepare(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(customer_features, pd.DataFrame):
            raise TypeError("Customer features must be supplied as a pandas DataFrame.")
        expected = set(self.feature_columns)
        received = set(customer_features.columns)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError(f"Feature schema mismatch ({'; '.join(details)}).")
        frame = customer_features.loc[:, list(self.feature_columns)].copy()
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Customer features must be finite numeric values.")
        if (frame < 0).any().any():
            raise ValueError("Customer features must be non-negative before log1p transformation.")
        return frame

    def transform(self, customer_features: pd.DataFrame) -> np.ndarray:
        frame = self._prepare(customer_features)
        return self.scaler.transform(np.log1p(frame.to_numpy(dtype=float)))

    def assign(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        matrix = self.transform(customer_features)
        anomaly_score = -self.anomaly_model.score_samples(matrix)
        return pd.DataFrame(
            {
                "cluster": self.kmeans.predict(matrix),
                "anomaly_score": anomaly_score,
                "is_anomaly": self.anomaly_model.predict(matrix) == -1,
            },
            index=customer_features.index,
        )
