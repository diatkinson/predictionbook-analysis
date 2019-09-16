import random
from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss

from nptyping import Array
from pba.features import calc_features
from pba.prediction import Prediction

random.seed(0)
Model = Any


class MeanCredencePredictor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, _):
        pass

    def predict(self, predictions: List[Prediction]) -> Array:
        y_pred = []
        for prediction in predictions:
            credences = prediction.credences()
            y_pred.append(np.mean(credences) / 100 if credences else 0.5)
        return np.array(y_pred)


class BaseRatePredictor(BaseEstimator):
    def __init__(self):
        self.base_rate = None

    def fit(self, predictions: List[Prediction]) -> None:
        self.base_rate = np.mean([prediction.right() for prediction in predictions])

    def predict(self, predictions: List[Prediction]) -> Array:
        return np.array([self.base_rate for _ in enumerate(predictions)])


class RandomPredictor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, _):
        pass

    def predict(self, predictions: List[Prediction]) -> Array:
        return np.random.random(len(predictions))


class FeaturePredictor(BaseEstimator):
    def __init__(self, relevant_features: Optional[List[str]] = None):
        self.predictor = LinearRegression()
        default_features = ["politics", "personal", "money", "negative_formulation",
                            "time_until_known", "avg_credence"]
        self.relevant_features = relevant_features or default_features

    def fit(self, predictions: List[Prediction]) -> None:
        feature_df = calc_features(predictions)
        self.predictor.fit(feature_df[self.relevant_features], feature_df['outcome'])

    def predict(self, predictions: List[Prediction]) -> None:
        return self.predictor.predict(calc_features(predictions)[self.relevant_features])


def evaluate(model: Model, train: List[Prediction], test: List[Prediction]) -> float:
    model.fit(train)
    y_true = np.array([int(pred.right()) for pred in test])
    y_pred = model.predict(test).clip(0, 1)
    return brier_score_loss(y_true, y_pred)
