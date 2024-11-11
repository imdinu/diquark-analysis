from abc import ABC, abstractmethod
import numpy as np
import awkward as ak

class BaseFeature(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compute(self, data: ak.Array) -> np.ndarray:
        """Compute the feature from the given data."""
        pass

    def __call__(self, data: ak.Array) -> np.ndarray:
        return self.compute(data)
    
    def _leading_jet_array(self, data: ak.Array, key: str, n_jets: int) -> ak.Array:
        """Extract the leading jet feature from the awkward array."""
        jet_pt_padded = ak.pad_none(data[key], n_jets, axis=-1, clip=True)
        return ak.to_numpy(ak.fill_none(jet_pt_padded, 0))