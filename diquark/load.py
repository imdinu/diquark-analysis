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


def read_jet_delphes(filename, library="ak"):
    """
    Read a delphes output TTree from a ROOT file into an awkward array.

    Args:
        filename (str): The path to the ROOT file containing the TTree.
        library (str): The name of the awkward array library to use (default is "ak").

    Returns:
        awkward.array: An awkward array containing the Jet, PT, Eta, Phi, and BTag branches from the TTree.
    """
    with uproot.open(filename) as f:
        tree = f["Delphes"]
        branches = filter_fbits(tree.typenames().keys())
        branches = ["Jet", "Jet/Jet.PT", "Jet/Jet.Eta", "Jet/Jet.Phi", "Jet/Jet.BTag"]
        return tree.arrays(branches, library=library)
    
    
def read_met_delphes(filename, library="ak"):
  
    with uproot.open(filename) as f:
        tree = f["Delphes"]
        branches2 = filter_fbits(tree.typenames().keys())
        branches2 = ["MissingET", "MissingET/MissingET.MET", "MissingET/MissingET.Eta", "MissingET/MissingET.Phi"]
        return tree.arrays(branches2, library=library)
