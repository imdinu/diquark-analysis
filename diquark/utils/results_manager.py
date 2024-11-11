import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

class ResultsManager:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: dict[str, Any], filename: str, custom_dir=None):
        file_path = (custom_dir if custom_dir else self.results_dir) / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def load_json(self, filename: str) -> dict[str, Any]:
        file_path = self.results_dir / filename
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_pickle(self, data: Any, filename: str):
        file_path = self.results_dir / filename
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filename: str) -> Any:
        file_path = self.results_dir / filename
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = 'csv', custom_dir=None):
        file_path = (custom_dir if custom_dir else self.results_dir) / filename
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_dataframe(self, filename: str, format: str = 'csv') -> pd.DataFrame:
        file_path = self.results_dir / filename
        if format == 'csv':
            return pd.read_csv(file_path)
        elif format == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_numpy(self, arr: np.ndarray, filename: str, custom_dir=None):
        file_path = (custom_dir if custom_dir else self.results_dir) / filename
        np.save(file_path, arr)

    def load_numpy(self, filename: str) -> np.ndarray:
        file_path = self.results_dir / filename
        return np.load(file_path)
    
    def create_subdir(self, subdir_name: str) -> Path:
        subdir = self.results_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)