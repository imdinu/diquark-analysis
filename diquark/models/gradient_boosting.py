import numpy as np
import xgboost as xgb
from diquark.models.base import BaseModel
from typing import Any

class GradientBoostingModel(BaseModel):
    def __init__(self, config: dict[str, Any]):
        super().__init__("GradientBoosting", config)
        self.n_estimators = self.config.get('n_estimators', 100)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.max_depth = self.config.get('max_depth', 3)
        self.subsample = self.config.get('subsample', 1.0)
        self.colsample_bytree = self.config.get('colsample_bytree', 1.0)
        self.random_state = self.config.get('random_state', 42)

    def build(self, input_shape: int):
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            tree_method="hist"
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        return self.model.evals_result()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model.load_model(path)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_