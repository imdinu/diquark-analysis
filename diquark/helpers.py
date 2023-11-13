import numpy as np
from scipy.optimize import curve_fit


def get_col(arr, col=0):
    """
    Returns the specified column of a 2D numpy array or the entire 1D array if it is already 1D.

    Args:
        arr (numpy.ndarray): The input array.
        col (int): The index of the column to return. Defaults to 0.

    Returns:
        numpy.ndarray: The specified column of the input array or the entire input array if it is already 1D.
    """
    return arr if len(arr.shape) == 1 else arr[:, col]


def create_data_dict(**kwargs):
    """
    Creates a dictionary of data arrays, where each key corresponds to a variable name and each value is an array of data
    for that variable. The function takes in keyword arguments, where each keyword corresponds to a variable name and each
    value is a dictionary of data arrays for that variable. The function concatenates the data arrays for each variable and
    returns a dictionary where each key corresponds to a variable name and each value is a concatenated array of data for
    that variable.

    Args:
        **kwargs (dict): A dictionary of keyword arguments, where each keyword corresponds to a variable name and each value
                     is a dictionary of data arrays for that variable.

    Returns:
        dict: A dictionary of data arrays, where each key corresponds to a variable name and each value is a concatenated
            array of data for that variable.
    """
    output = {"Truth": []}
    for var, data in kwargs.items():
        for key, arr in data.items():
            n = len(arr.shape)
            truth_value = np.repeat(key, arr.shape[0])
            if n > 1:
                for i in range(arr.shape[1]):
                    x = get_col(arr, i)
                    o_key = f"{var}_{i+1}"
                    if o_key not in output:
                        output[o_key] = []
                    output[o_key].append(x)
            else:
                x = arr
                o_key = f"{var}"
                if o_key not in output:
                    output[o_key] = []
                output[o_key].append(x)
            if o_key == "multiplicity":
                output["Truth"].append(truth_value)

    return {k: np.concatenate(v) for k, v in output.items()}


def mass_score_cut(masses, scores, cut=0.5, prc=True):
    """
    Filter masses based on score cutoff.

    Args:
        masses (dict): A dictionary of lables and arrays of masses.
        scores (dict): A dictionary of lables and classification scores.
        cut (float, optional): The score cutoff. Defaults to 0.5.
        prc (bool, optional): Whether to interpret `cut` as a percentile. Defaults to True.

    Returns:
        dict: A dictionary of filtered masses.
    """
    output = {}
    for key in masses.keys():
        if prc:
            score_cut = np.percentile(scores[key], cut*100)
        else:
            score_cut = cut
        output[key] = masses[key][scores[key] > score_cut]
    return output


def gaussian(x, mean, amplitude, standard_deviation):
    """
    Returns the value of a Gaussian function at a given point x.

    Args:
        x (float): The point at which to evaluate the Gaussian function.
        mean (float): The mean of the Gaussian distribution.
        amplitude (float): The amplitude of the Gaussian function.
        standard_deviation (float): The standard deviation of the Gaussian distribution.

    Returns:
        float: The value of the Gaussian function at the given point x.
    """
    return amplitude * np.exp( - ((x - mean) ** 2 / (2 * standard_deviation ** 2)))


def build_gaussian(mean):
    """
    Returns a Gaussian function with the given mean.

    Parameters:
    mean (float): The mean of the Gaussian function.

    Returns:
    function: A Gaussian function that takes in an x value, amplitude, and standard deviation.
    """
    def gaussian(x, amplitude, standard_deviation):
        return amplitude * np.exp( - ((x - mean) ** 2 / (2 * standard_deviation ** 2)))
    return gaussian