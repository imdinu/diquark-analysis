import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from diquark.models.base import BaseModel
from typing import Any

class NeuralNetworkModel(BaseModel):
    def __init__(self, config: dict[str, Any]):
        super().__init__("NeuralNetwork", config)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 128)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.layer_sizes = self.config.get('layer_sizes', [64, 32, 32])
        self.dropout_rates = self.config.get('dropout_rates', [0.2, 0.1])

    def build(self, input_shape: int):
        model_layers = [layers.InputLayer(input_shape=(input_shape,))]
        
        for i, size in enumerate(self.layer_sizes):
            model_layers.append(layers.Dense(size, activation="relu"))
            if i < len(self.dropout_rates):
                model_layers.append(layers.Dropout(self.dropout_rates[i]))
        
        model_layers.append(layers.Dense(1, activation="sigmoid"))
        
        self.model = models.Sequential(model_layers)
        
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).flatten()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = models.load_model(path)