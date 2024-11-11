from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class BaseModel(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model: Any = None

    @abstractmethod
    def build(self, input_shape: int):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the trained model."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load a trained model."""
        pass