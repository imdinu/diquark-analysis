from itertools import combinations

import numpy as np
import awkward as ak
from scipy.spatial.distance import cdist


def jet_multiplicity(arr):
    """
    Returns the number of jets in the input array.

    Args:
        arr (awkward.Array): Input array containing jet information.

    Returns:
        numpy.ndarray: Array containing the number of jets in each event.
    """
    return ak.to_numpy(arr["Jet"])


def leading_jet_arr(arr, n=6, key="Jet/Jet.PT"):
    """
    Returns an array containing the leading jet transverse momentum (PT) values.

    Args:
        arr (awkward.Array): Input array containing jet information.
        n (int, optional): Number of leading jets to return. Defaults to 6.
        key (str, optional): Key to access the jet PT values. Defaults to "Jet/Jet.PT".

    Returns:
        numpy.ndarray: Array containing the leading jet PT values.
    """
    jet_pt_padded = ak.pad_none(arr[key], n, axis=-1, clip=True)
    return ak.to_numpy(ak.fill_none(jet_pt_padded, 0))


def calculate_delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """
    Calculates the element-wise difference in angle between two given arrays of angles, phi1 and phi2.
    The result is wrapped to the range [-pi, pi].

    Args:
        phi1 (numpy.ndarray): Array of angles.
        phi2 (numpy.ndarray): Array of angles.

    Returns:
        numpy.ndarray: Array of element-wise difference in angle between phi1 and phi2, wrapped to the range [-pi, pi].
    """

    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)

    return dphi


def calculate_delta_r(etas, phis, pts):
    """
    Calculates the pairwise ΔR distance between jets in an event.

    Args:
        etas (numpy.ndarray): Array of shape (n_events, n_jets) containing the pseudorapidity
            values of each jet in each event.
        phis (numpy.ndarray): Array of shape (n_events, n_jets) containing the azimuthal angle
            values of each jet in each event.
        pts (numpy.ndarray): Array of shape (n_events, n_jets) containing the transverse momentum
            values of each jet in each event.

    Returns:
        numpy.ndarray: Array of shape (n_events, n_pairs) containing the ΔR distance between each
            pair of jets in each event, where n_pairs = n_jets * (n_jets - 1) // 2.
    """
    n_events, n_jets = etas.shape
    n_pairs = n_jets * (n_jets - 1) // 2

    # Expand dimensions to create a pairwise difference matrix
    delta_eta = etas[:, :, None] - etas[:, None, :]
    delta_phi = calculate_delta_phi(phis[:, :, None], phis[:, None, :])

    # Filter out differences between the same jet
    delta_eta = np.triu(delta_eta, k=1)
    delta_phi = np.triu(delta_phi, k=1)

    # Combine the pairwise differences into a ΔR matrix
    delta_r_matrix = np.sqrt(delta_eta**2 + delta_phi**2)

    # Create a mask with False on the diagonal and True everywhere else
    pts_mask = np.ones((n_events, n_jets, n_jets), dtype=bool)
    for i in range(n_events):
        np.fill_diagonal(pts_mask[i], 0)
    pts_mask &= pts[:, :, None] * pts[:, None, :] > 0

    # Apply the mask to the ΔR matrix
    delta_r_matrix *= pts_mask

    # Reshape the ΔR matrix into a 2D array of shape (n_events, n_pairs)
    delta_r_array = delta_r_matrix.reshape(n_events, -1)[:, :n_pairs]

    return delta_r_array


def combined_invariant_mass(arr, n=6):
    """
    Calculates the combined invariant mass of the leading n jets in the given array.

    Parameters:
        arr (awkward.Array): Awkward array of jets.
        n (int): Number of leading jets to consider. Default is 6.

    Returns:
        numpy.ndarray: Array of combined invariant masses.
    """
    E_total = np.sum(
        leading_jet_arr(arr, n, "Jet/Jet.PT") * np.cosh(leading_jet_arr(arr, n, "Jet/Jet.Eta")),
        axis=1,
    )
    px_total = np.sum(
        leading_jet_arr(arr, n, "Jet/Jet.PT") * np.cos(leading_jet_arr(arr, n, "Jet/Jet.Phi")),
        axis=1,
    )
    py_total = np.sum(
        leading_jet_arr(arr, n, "Jet/Jet.PT") * np.sin(leading_jet_arr(arr, n, "Jet/Jet.Phi")),
        axis=1,
    )
    pz_total = np.sum(
        leading_jet_arr(arr, n, "Jet/Jet.PT") * np.sinh(leading_jet_arr(arr, n, "Jet/Jet.Eta")),
        axis=1,
    )

    return np.nan_to_num(np.sqrt(E_total**2 - px_total**2 - py_total**2 - pz_total**2))


def three_jet_invariant_mass(arr, n=6):
    """
    Computes the invariant mass of n choose 3 combinations of the leading n jets in each event.

    Parameters:
        arr (awkward.Array): Array of jet features.
        n (int): Number of jets to consider in each event.

    Returns:
        numpy.ndarray: Array of invariant masses for each event.
    """
    # Retrieve jet features
    pts = leading_jet_arr(arr, n, "Jet/Jet.PT")
    etas = leading_jet_arr(arr, n, "Jet/Jet.Eta")
    phis = leading_jet_arr(arr, n, "Jet/Jet.Phi")
    
    # Compute invariant mass for each event
    masses = []
    for event_pts, event_etas, event_phis in zip(pts, etas, phis):
        event_masses = []
        for i, j, k in combinations(range(n), 3): # Combinations of 3 jets
            # If any jet has pt 0, set combined mass to 0
            if event_pts[i] == 0 or event_pts[j] == 0 or event_pts[k] == 0:
                mass = 0
            else:
                # Compute energy, px, py, and pz for the 3 jets
                E = sum(event_pts[idx] * np.cosh(event_etas[idx]) for idx in (i, j, k))
                px = sum(event_pts[idx] * np.cos(event_phis[idx]) for idx in (i, j, k))
                py = sum(event_pts[idx] * np.sin(event_phis[idx]) for idx in (i, j, k))
                pz = sum(event_pts[idx] * np.sinh(event_etas[idx]) for idx in (i, j, k))
                # Compute invariant mass for the 3 jets
                mass = np.nan_to_num(np.sqrt(E**2 - px**2 - py**2 - pz**2))
            event_masses.append(mass)
        # Sort masses in descending order for each event
        masses.append(sorted(event_masses, reverse=True))
    return np.array(masses)
