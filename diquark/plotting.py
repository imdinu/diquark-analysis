import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go

from .helpers import get_col, gaussian, build_gaussian, mass_score_cut
from . import COLOR_DICT, CROSS_SECTION_DICT


def make_histogram(
    variable,
    nbins,
    col=0,
    color=COLOR_DICT,
    cross=CROSS_SECTION_DICT,
    clip_top_prc=99.9,
    clip_bottom_prc=0.1,
):
    """
    Create a plot of multiple labeled histograms for a given variable.

    Args:
        variable (dict): A dictionary of lables and their corresponding data.
        nbins (int): The number of bins to use in the histogram.
        col (int, optional): The column index to use for the data. Defaults to 0.
        color (dict, optional): A dictionary of lables and their corresponding colors. Defaults to COLOR_DICT.
        cross (dict, optional): A dictionary of lables and their corresponding cross sections. Defaults to CROSS_SECTION_DICT.
        clip_top_prc (float, optional): The percentile of data to clip from the top. Defaults to 99.9.
        clip_bottom_prc (float, optional): The percentile of data to clip from the bottom. Defaults to 0.1.

    Returns:
        plotly.graph_objs._figure.Figure: A histogram plot of the given variable.
    """
    data_cols = {key: get_col(data, col) for key, data in variable.items()}
    all_values = np.concatenate(list(data_cols.values()))
    min_val = np.percentile(all_values, clip_bottom_prc)
    max_val = np.percentile(all_values, clip_top_prc)
    _, bin_edges = np.histogram(
        np.clip(all_values, min_val, max_val),
        nbins,
    )
    fig = go.Figure()
    for key, data in data_cols.items():
        counts, _ = np.histogram(np.clip(data, min_val, max_val), bins=bin_edges, density=not cross)
        fig.add_trace(
            go.Bar(
                x=bin_edges,
                y=counts * (cross[key] if cross else 1),
                name=key,
                marker_color=color[key],
            ),
        )
    return fig


def make_histogram_with_double_gaussian_fit(
    variable,
    nbins,
    col=0,
    color=COLOR_DICT,
    cross=CROSS_SECTION_DICT,
    clip_top_prc=99.9,
    clip_bottom_prc=0.1,
):
    """
    Creates a histogram with a double Gaussian fit for the given variable.

    Args:
        variable (dict): A dictionary of data arrays to plot.
        nbins (int): The number of bins to use in the histogram.
        col (int, optional): The column index to use for the data. Defaults to 0.
        color (dict, optional): A dictionary of colors to use for each data array. Defaults to COLOR_DICT.
        cross (dict, optional): A dictionary of cross sections to use for each data array. Defaults to CROSS_SECTION_DICT.
        clip_top_prc (float, optional): The percentage of data to clip from the top. Defaults to 99.9.
        clip_bottom_prc (float, optional): The percentage of data to clip from the bottom. Defaults to 0.1.

    Returns:
        fig (go.Figure): A plotly figure object containing the histogram and Gaussian fit.
    """
    data_cols = {key: get_col(data, col) for key, data in variable.items()}
    all_values = np.concatenate(list(data_cols.values()))
    min_val = np.percentile(all_values, clip_bottom_prc)
    max_val = np.percentile(all_values, clip_top_prc)
    _, bin_edges = np.histogram(
        np.clip(all_values, min_val, max_val),
        nbins,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    aggregated_counts = np.zeros_like(bin_centers)

    fig = go.Figure()
    for key, data in data_cols.items():
        counts, _ = np.histogram(np.clip(data, min_val, max_val), bins=bin_edges, density=not cross)
        counts_scaled = counts * (cross[key] if cross else 1)
        aggregated_counts += counts_scaled
        if counts.sum() == 0:
            continue
        fig.add_trace(
            go.Bar(x=bin_edges, y=counts_scaled, name=key, marker_color=color[key]),
        )

    # First Gaussian fit
    params, _ = curve_fit(
        gaussian,
        bin_centers,
        aggregated_counts,
        p0=[
            bin_centers[np.argmax(aggregated_counts)],
            max(aggregated_counts),
            (bin_centers[-1] - bin_centers[0]) / 2,
        ],
    )
    mean1, amplitude1, std_dev1 = params
    std_dev1 = abs(std_dev1)

    # Second Gaussian fit with the interval (x ± 2σx)
    mask = (bin_centers > mean1 - 2 * std_dev1) & (bin_centers < mean1 + 2 * std_dev1)
    params2, _ = curve_fit(
        build_gaussian(mean1),
        bin_centers[mask],
        aggregated_counts[mask],
        p0=(amplitude1, std_dev1),
    )
    amplitude2, std_dev2 = params2

    # Adding the fit to the plot
    x_fit = np.linspace(min_val, max_val, 1000)
    y_fit2 = gaussian(x_fit, mean1, amplitude2, std_dev2)
    fig.add_trace(
        go.Scatter(
            x=x_fit, y=y_fit2, mode="lines", name=f"Fit: mean={mean1:.3f}, std_dev={std_dev2:.3f}"
        )
    )

    return fig
