import uproot


def filter_fbits(branches):
    """
    Filter out branch names containing 'fBits'.

    Args:
        branches (list): List of branch names.

    Returns:
        list: List of branch names that do not contain 'fBits'.
    """
    return [b for b in branches if "fBits" not in b]


def read_jet_delphes(
    filename,
    library="ak",
    branches=[
        "Jet",
        "Jet/Jet.PT",
        "Jet/Jet.Eta",
        "Jet/Jet.Phi",
        "Jet/Jet.BTag",
        "Particle/Particle.PID",
        "Particle/Particle.Status",
        "Particle/Particle.Mass",
    ],
):
    """
    Read a delphes output TTree from a ROOT file into an awkward array.

    Args:
        filename (str): The path to the ROOT file containing the TTree.
        library (str): The name of the awkward array library to use (default is "ak").
        branches (list): A list of branches to read from the TTree (default is ["Jet", "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.BTag"]).
    Returns:
        awkward.array: An awkward array containing the Jet, PT, Eta, Phi, and BTag branches from the TTree.
    """
    with uproot.open(filename) as f:
        tree = f["Delphes"]
        branches = filter_fbits(branches)
        return tree.arrays(branches, library=library)


def lower_cut_suu_mass(arr, mass):
    """
    Filter out jets with mass less than the given value.

    Args:
        arr (awkward.array): An awkward array containing the truth branches.
        mass (float): The minimum mass value to filter on.

    Returns:
        awkward.array: An awkward array containing the Jet branch with mass greater than the given value.
    """
    mask = (arr["Particle/Particle.PID"] == 9936661) & (arr["Particle/Particle.Status"] == 22)
    masses = arr["Particle/Particle.Mass"][mask].to_numpy()
    print((masses >= mass).sum() / len(masses))
    return arr[(masses > mass).flatten()]
