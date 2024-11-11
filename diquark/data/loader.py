import uproot
import awkward as ak
import numpy as np

from tqdm import tqdm

from diquark.config.constants import DATA_KEYS

class DataLoader:

    def __init__(self, path_dict: dict[str, str], n_jets=6, index_start=0, index_stop=None):
        self.default_branches = [
            "Jet",
            "Jet/Jet.PT",
            "Jet/Jet.Eta",
            "Jet/Jet.Phi",
            "Jet/Jet.BTag",
            "Particle/Particle.PID",
            "Particle/Particle.Status",
            "Particle/Particle.Mass",
        ]
        self.path_dict = path_dict
        self.n_jets = n_jets
        self.index_start = index_start
        self.index_stop = index_stop

    def filter_fbits(self, branches: list[str]) -> list[str]:
        """Filter out branch names containing 'fBits'."""
        return [b for b in branches if "fBits" not in b]

    def read_jet_delphes(self, filename: str, branches: list[str] = None) -> ak.Array:
        """Read a delphes output TTree from a ROOT file into an awkward array."""
        if branches is None:
            branches = self.default_branches

        with uproot.open(filename) as f:
            tree = f["Delphes"]
            branches = self.filter_fbits(branches)
            return tree.arrays(branches, library="ak", entry_start=self.index_start, entry_stop=self.index_stop)

    def lower_cut_suu_mass(self, arr: ak.Array, mass: float) -> ak.Array:
        """Filter out jets with mass less than the given value."""
        mask = (arr["Particle/Particle.PID"] == 9936661) & (arr["Particle/Particle.Status"] == 22)
        masses = arr["Particle/Particle.Mass"][mask].to_numpy()
        print(f"Fraction of events passing mass cut: {(masses >= mass).sum() / len(masses):.2f}")
        return arr[(masses > mass).flatten()]

    def load_data(self, mass_cut: float = None) -> dict[str, ak.Array]:
        """Load all datasets specified in DATA_KEYS."""
        self.datasets = {}
        for key in tqdm(DATA_KEYS, desc="Loading data"):
            arr = self.read_jet_delphes(self.path_dict[key])
            if key.startswith("SIG") and mass_cut is not None:
                arr = self.lower_cut_suu_mass(arr, mass_cut)
            self.datasets[key] = arr
        return self.datasets


def get_jet_features(arr: ak.Array, n_jets) -> dict[str, np.ndarray]:
    """Extract basic jet features from the awkward array."""
    return {
        "pt": ak.to_numpy(ak.pad_none(arr["Jet/Jet.PT"], n_jets, clip=True)),
        "eta": ak.to_numpy(ak.pad_none(arr["Jet/Jet.Eta"], n_jets, clip=True)),
        "phi": ak.to_numpy(ak.pad_none(arr["Jet/Jet.Phi"], n_jets, clip=True)),
        "btag": ak.to_numpy(ak.pad_none(arr["Jet/Jet.BTag"], n_jets, clip=True)),
    }