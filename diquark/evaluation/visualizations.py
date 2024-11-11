import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any
from .metrics import calculate_signal_background_metrics
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_results(results: dict[str, Any], df_test: pd.DataFrame, plot_types: list[str], real_percentiles) -> dict[str, go.Figure]:
    """
    Create various plots based on model results.
    
    Args:
        results: Dictionary containing model results (metrics, predictions, etc.).
        df_test: Test dataset as a pandas DataFrame.
        plot_types: List of plot types to generate.
    
    Returns:
        Dictionary of plotly figures.
    """
    plots = {}
    
    if 'roc_curve' in plot_types:
        fpr, tpr, _ = roc_curve(df_test['target'], results['predictions'])
        plots['roc_curve'] = plot_roc_curve(fpr, tpr, results['metrics']['roc_auc'])
    
    if 'pr_curve' in plot_types:
        precision, recall, _ = precision_recall_curve(df_test['target'], results['predictions'])
        plots['pr_curve'] = plot_precision_recall_curve(recall, precision, results['metrics']['average_precision'])
    
    if 'weighted_pr_curve' in plot_types:
        plots['weighted_pr_curve'] = plot_weighted_precision_recall_curve(
            results['metrics']['weighted_recall'],
            results['metrics']['weighted_precision'],
            results['metrics']['weighted_pr_thresholds'],
            results['metrics']['weighted_pr_auc'],
            use_real_event_percentiles=real_percentiles
        )

    if 'feature_importances' in plot_types and 'feature_importances' in results:
        plots['feature_importances'] = plot_feature_importances(df_test.columns[:-2], results['feature_importances'])
    
    if 'sig_bkg_metrics' in plot_types and 'sig_bkg_metrics' in results:
        plots['sig_bkg_metrics'] = plot_signal_background_metrics(results['sig_bkg_metrics'], real_percentiles)
    
    return plots

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random classifier', line=dict(dash='dash')))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=600
    )
    return fig

def plot_precision_recall_curve(recall: np.ndarray, precision: np.ndarray, average_precision: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve (AP = {average_precision:.3f})'))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=800, height=600
    )
    return fig

def plot_weighted_precision_recall_curve(recall: np.ndarray, precision: np.ndarray, 
                                         thresholds: np.ndarray, pr_auc: float, 
                                         use_real_event_percentiles: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, 
        mode='lines', 
        name=f'Weighted PR curve (AUC = {pr_auc:.3f})',
        text=[f'Threshold: {t:.3f}' for t in thresholds],
        hoverinfo='text+x+y'
    ))
    fig.update_layout(
        title='Weighted Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=800, height=600
    )
    if use_real_event_percentiles:
        fig.update_layout(
            xaxis_title='Recall (real event percentiles)',
            yaxis_title='Precision (real event percentiles)'
        )
    return fig

def plot_feature_importances(feature_names: np.ndarray, importances: np.ndarray) -> go.Figure:
    sorted_idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        y=feature_names[sorted_idx],
        x=importances[sorted_idx],
        orientation='h'
    ))
    fig.update_layout(
        title='Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        width=800, height=len(feature_names) * 20 + 200
    )
    return fig

def plot_signal_background_metrics(metrics: dict[str, np.ndarray], use_real_event_percentiles: bool) -> go.Figure:
    thresholds = metrics['thresholds']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=thresholds, y=metrics['signal_efficiency'], name='Signal Efficiency'), row=1, col=1)
    fig.add_trace(go.Scatter(x=thresholds, y=metrics['background_rejection'], name='Background Rejection'), row=1, col=1)
    fig.add_trace(go.Scatter(x=thresholds, y=metrics['significance'], name='Significance'), row=2, col=1)
    fig.add_trace(go.Scatter(x=thresholds, y=metrics['s_over_b'], name='S/B'), row=3, col=1)
    fig.update_layout(
        title='Signal-Background Metrics',
        xaxis_title='Threshold' if not use_real_event_percentiles else 'Percentile (real events)',
        width=800, height=1000
    )
    fig.update_yaxes(title_text="Efficiency/Rejection", row=1, col=1)
    fig.update_yaxes(title_text="Significance", row=2, col=1)

    return fig

