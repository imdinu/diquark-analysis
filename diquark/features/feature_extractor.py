import numpy as np
import awkward as ak
from itertools import combinations

class FeatureExtractor:
    def __init__(self, n_jets: int):
        self.n_jets = n_jets
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self) -> list[str]:
        basic_features = [
            "jet_multiplicity",
            *[f"leading_jet_pt_{i+1}" for i in range(self.n_jets)],
            *[f"leading_jet_eta_{i+1}" for i in range(self.n_jets)],
            *[f"leading_jet_phi_{i+1}" for i in range(self.n_jets)],
            *[f"delta_r_{i+1}_{j+1}" for i in range(self.n_jets) for j in range(i+1, self.n_jets)],
            "combined_invariant_mass",
        ]
        
        for k in [2, 3]:
            basic_features.extend([f"{k}jet_invariant_mass_{i+1}" for i in range(self.n_jets)])
            basic_features.extend([f"{k}jet_vector_sum_pt_{i+1}" for i in range(self.n_jets)])
        
        combined_features = [
            "m3j_m6j_ratio",
            "m2j_m6j_ratio",
            "n_jet_pairs_near_w_mass",
            "max_delta_r",
            "smallest_delta_r_mass",
            "max_vector_sum_pt",
        ]
        
        return basic_features + combined_features

    def _leading_jet_array(self, data: ak.Array, key: str) -> np.ndarray:
        jet_pt_padded = ak.pad_none(data[key], self.n_jets, axis=-1, clip=True)
        return ak.to_numpy(ak.fill_none(jet_pt_padded, 0))

    def jet_multiplicity(self, data: ak.Array) -> np.ndarray:
        return ak.to_numpy(data["Jet"])

    def leading_jet_pt(self, data: ak.Array) -> np.ndarray:
        return self._leading_jet_array(data, "Jet/Jet.PT")

    def leading_jet_eta(self, data: ak.Array) -> np.ndarray:
        return self._leading_jet_array(data, "Jet/Jet.Eta")

    def leading_jet_phi(self, data: ak.Array) -> np.ndarray:
        return self._leading_jet_array(data, "Jet/Jet.Phi")

    def delta_r(self, data: ak.Array) -> np.ndarray:
        etas = self.leading_jet_eta(data)
        phis = self.leading_jet_phi(data)
        pts = self.leading_jet_pt(data)

        n_events, _ = etas.shape
        n_pairs = self.n_jets * (self.n_jets - 1) // 2

        delta_eta = etas[:, :, None] - etas[:, None, :]
        delta_phi = self._calculate_delta_phi(phis[:, :, None], phis[:, None, :])

        delta_r_matrix = np.sqrt(delta_eta**2 + delta_phi**2)
        delta_r_matrix = np.triu(delta_r_matrix, k=1)

        pts_mask = np.ones((n_events, self.n_jets, self.n_jets), dtype=bool)
        for i in range(n_events):
            np.fill_diagonal(pts_mask[i], 0)
        pts_mask &= pts[:, :, None] * pts[:, None, :] > 0

        delta_r_matrix *= pts_mask
        delta_r_array = delta_r_matrix.reshape(n_events, -1)[:, :n_pairs]

        return delta_r_array

    @staticmethod
    def _calculate_delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        dphi = phi1 - phi2
        dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
        return dphi

    def combined_invariant_mass(self, data: ak.Array) -> np.ndarray:
        pt = self.leading_jet_pt(data)
        eta = self.leading_jet_eta(data)
        phi = self.leading_jet_phi(data)

        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = pt * np.cosh(eta)

        px_total = ak.sum(px, axis=1)
        py_total = ak.sum(py, axis=1)
        pz_total = ak.sum(pz, axis=1)
        E_total = ak.sum(E, axis=1)

        mass = np.sqrt(E_total**2 - px_total**2 - py_total**2 - pz_total**2)
        return ak.to_numpy(mass)

    def n_jet_invariant_mass(self, data: ak.Array, k: int) -> np.ndarray:
        pt = self.leading_jet_pt(data)
        eta = self.leading_jet_eta(data)
        phi = self.leading_jet_phi(data)

        masses = []
        for event_pt, event_eta, event_phi in zip(pt, eta, phi):
            event_masses = []
            for indices in combinations(range(self.n_jets), k):
                if any(event_pt[i] == 0 for i in indices):
                    mass = 0
                else:
                    E = sum(event_pt[idx] * np.cosh(event_eta[idx]) for idx in indices)
                    px = sum(event_pt[idx] * np.cos(event_phi[idx]) for idx in indices)
                    py = sum(event_pt[idx] * np.sin(event_phi[idx]) for idx in indices)
                    pz = sum(event_pt[idx] * np.sinh(event_eta[idx]) for idx in indices)
                    mass = np.nan_to_num(np.sqrt(E**2 - px**2 - py**2 - pz**2))
                event_masses.append(mass)
            masses.append(sorted(event_masses, reverse=True))

        return np.array(masses)

    def n_jet_vector_sum_pt(self, data: ak.Array, k: int) -> np.ndarray:
        pts = self.leading_jet_pt(data)
        phis = self.leading_jet_phi(data)
        vector_sum_pts = []
        for event_pts, event_phis in zip(pts, phis):
            vector_sums = []
            for indices in combinations(range(self.n_jets), k):
                if any(event_pts[i] == 0 for i in indices):
                    vector_sum_pt = 0
                else:
                    px = sum(event_pts[idx] * np.cos(event_phis[idx]) for idx in indices)
                    py = sum(event_pts[idx] * np.sin(event_phis[idx]) for idx in indices)
                    vector_sum_pt = np.sqrt(px**2 + py**2)
                vector_sums.append(vector_sum_pt)
            vector_sum_pts.append(sorted(vector_sums, reverse=True))
        return np.array(vector_sum_pts)

    def flatten_features(self, features: dict[str, np.ndarray]) -> np.ndarray:
        flat_features = {}
        for feature, values in features.items():
            match values.ndim:
                case 1:
                    flat_features[feature] = values
                case 2:
                    for i in range(values.shape[1]):
                        flat_features[f"{feature}_{i+1}"] = values[:, i]
                case _:
                    raise ValueError(f"Invalid feature shape: {values.shape}")
        return flat_features

    def compute_all(self, data: ak.Array) -> dict[str, np.ndarray]:
        features = {
            "jet_multiplicity": self.jet_multiplicity(data),
            "leading_jet_pt": self.leading_jet_pt(data),
            "leading_jet_eta": self.leading_jet_eta(data),
            "leading_jet_phi": self.leading_jet_phi(data),
            "delta_r": self.delta_r(data),
            "combined_invariant_mass": self.combined_invariant_mass(data),
        }

        for k in [2, 3]:
            features[f"{k}jet_invariant_mass"] = self.n_jet_invariant_mass(data, k)
            features[f"{k}jet_vector_sum_pt"] = self.n_jet_vector_sum_pt(data, k)

        # Compute combined features
        mnj = features["combined_invariant_mass"]
        m3j = features["3jet_invariant_mass"]
        m2j = features["2jet_invariant_mass"]

        features[f"m3j_m{self.n_jets}j_ratio"] = np.divide(
            m3j.mean(axis=1, where=m3j != 0),
            mnj,
            out=np.zeros_like(mnj),
            where=mnj != 0
        )

        features[f"m2j_m{self.n_jets}j_ratio"] = np.divide(
            m2j.mean(axis=1, where=m2j != 0),
            mnj,
            out=np.zeros_like(mnj),
            where=mnj != 0
        )

        features["n_jet_pairs_near_w_mass"] = np.sum((m2j >= 60) & (m2j <= 100), axis=1)
        features["max_delta_r"] = np.max(features["delta_r"], axis=1)

        smallest_delta_r_indices = np.argmin(features["delta_r"], axis=1)
        features["smallest_delta_r_mass"] = np.choose(smallest_delta_r_indices, m2j.T)

        features["max_vector_sum_pt"] = np.max(features["2jet_vector_sum_pt"], axis=1)
        features.pop("3jet_vector_sum_pt")
        features.pop("2jet_vector_sum_pt")

        return self.flatten_features(features)

