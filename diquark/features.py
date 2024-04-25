from itertools import combinations
from warnings import warn

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


def jet_btag_multiplicity(arr):
    """
    Returns the number of b-tagged jets in the input array.

    Args:
        arr (awkward.Array): Input array containing jet information.

    Returns:
        numpy.ndarray: Array containing the number of b-tagged jets in each event.
    """
    return ak.to_numpy(ak.sum(arr["Jet/Jet.BTag"], axis=1))


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


def n_jet_vector_sum_pt(arr, n, k):
    """
    Computes the vector sum of transverse momentum (pT) for n choose k combinations of the leading n jets in each event.

    Parameters:
        arr (awkward.Array): Array of jet features.
        n (int): Number of jets to consider in each event.
        k (int): Number of jets to combine for vector sum pT calculation.

    Returns:
        numpy.ndarray: Array of vector sum pT for each event.
    """
    # Retrieve jet features
    pts = leading_jet_arr(arr, n, "Jet/Jet.PT")
    phis = leading_jet_arr(arr, n, "Jet/Jet.Phi")

    # Compute vector sum pT for each event
    vector_sums_pt = []
    for event_pts, event_phis in zip(pts, phis):
        event_vector_sums = []
        for indices in combinations(range(n), k): # Combinations of k jets
            # Check if any jet in the combination has pt 0
            if any(event_pts[i] == 0 for i in indices):
                vector_sum_pt = 0
            else:
                # Compute x and y components of pT for the k jets
                px = sum(event_pts[idx] * np.cos(event_phis[idx]) for idx in indices)
                py = sum(event_pts[idx] * np.sin(event_phis[idx]) for idx in indices)
                # Compute vector sum pT for the k jets
                vector_sum_pt = np.sqrt(px**2 + py**2)
            event_vector_sums.append(vector_sum_pt)
        # Sort vector sums in descending order for each event
        vector_sums_pt.append(sorted(event_vector_sums, reverse=True))
    return np.array(vector_sums_pt)


def n_jet_invariant_mass(arr, n, k):
    """
    Computes the invariant mass of n choose k combinations of the leading n jets in each event.

    Parameters:
        arr (awkward.Array): Array of jet features.
        n (int): Number of jets to consider in each event.
        k (int): Number of jets to combine for invariant mass calculation.

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
        for indices in combinations(range(n), k): # Combinations of k jets
            # Check if any jet in the combination has pt 0
            if any(event_pts[i] == 0 for i in indices):
                mass = 0
            else:
                # Compute energy and momentum components for the k jets
                E = sum(event_pts[idx] * np.cosh(event_etas[idx]) for idx in indices)
                px = sum(event_pts[idx] * np.cos(event_phis[idx]) for idx in indices)
                py = sum(event_pts[idx] * np.sin(event_phis[idx]) for idx in indices)
                pz = sum(event_pts[idx] * np.sinh(event_etas[idx]) for idx in indices)
                # Compute invariant mass for the k jets
                mass = np.nan_to_num(np.sqrt(E**2 - px**2 - py**2 - pz**2))
            event_masses.append(mass)
        # Sort masses in descending order for each event
        masses.append(sorted(event_masses, reverse=True))
    return np.array(masses)


def n_jet_charge_sum(arr, n, k):
    """
    Computes the sum of charges for n choose k combinations of the leading n jets in each event.

    Parameters:
        arr (awkward.Array): Array of jet features.
        n (int): Number of jets to consider in each event.
        k (int): Number of jets to combine for charge sum calculation.

    Returns:
        numpy.ndarray: Array of charge sums for each event.
    """
    # Retrieve jet features
    charges = leading_jet_arr(arr, n, "Jet/Jet.Charge")
    
    # Compute charge sum for each event
    charge_sums = []
    for event_charges in charges:
        event_charge_sums = []
        for indices in combinations(range(n), k): # Combinations of k jets
            # Compute charge sum for the k jets
            charge_sum = sum(event_charges[idx] for idx in indices)
            event_charge_sums.append(charge_sum)
        # Sort charge sums in descending order for each event
        charge_sums.append(sorted(event_charge_sums, reverse=True))
    return np.array(charge_sums)


def three_jet_invariant_mass(arr, n=6):
    warn(
        "three_jet_invariant_mass is deprecated and will be removed in a future release. Use n_jet_invariant_mass instead.",
        DeprecationWarning,
    )
    return n_jet_invariant_mass(arr, n, 3)