"""
HRP.py

Hierarchical Risk Parity — standalone functions for portfolio weight allocation.

Given a returns DataFrame (dates × tickers), computes risk-balanced weights
using correlation clustering and recursive bisection.

No model predictions are used here — this is purely a risk allocation tool.
Selection happens elsewhere (your model). HRP determines how to distribute
capital among the selected stocks.

Reference: López de Prado, "Building Diversified Portfolios that Outperform
Out-of-Sample" (2016).

Usage:
    from HRP import compute_hrp_weights

    # returns_df: DataFrame with dates as index, tickers as columns, log returns as values
    weights = compute_hrp_weights(returns_df)
    # weights = {"AAPL": 0.12, "MSFT": 0.08, "XOM": 0.15, ...}
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def compute_hrp_weights(returns_df, linkage_method="single"):
    """
    Main entry point. Takes a returns DataFrame and returns HRP weights.

    Args:
        returns_df:      DataFrame with dates as index, tickers as columns,
                         daily log returns as values.
        linkage_method:  Clustering method — 'single', 'complete', 'average', or 'ward'.

    Returns:
        dict of {ticker: weight}. All positive, sum to 1.0.
        Falls back to equal weight if inputs are degenerate.
    """
    # Clean inputs
    returns_df = returns_df.dropna(axis=1, how="all")  # drop tickers with no data
    returns_df = returns_df.dropna(axis=0, how="any")  # drop days with any missing

    # Drop zero-variance columns (constant price stocks)
    variances = returns_df.var()
    zero_var = variances[variances < 1e-12].index.tolist()
    if zero_var:
        returns_df = returns_df.drop(columns=zero_var)

    tickers = list(returns_df.columns)
    n = len(tickers)

    # Edge cases: fall back to equal weight
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}
    if n <= 1 or len(returns_df) < 10:
        w = 1.0 / n
        return {t: w for t in tickers}

    # Step 1: Correlation and covariance
    corr = returns_df.corr()
    cov = returns_df.cov()

    # Regularize: add small value to diagonal to prevent singular matrix
    cov_values = cov.values.copy()
    np.fill_diagonal(cov_values, np.diag(cov_values) + 1e-8)
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)

    # Step 2: Correlation distance
    dist = correlation_distance(corr)

    # Step 3: Hierarchical clustering
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method=linkage_method)

    # Step 4: Quasi-diagonalize
    sorted_tickers = quasi_diagonalize(link, tickers)
    sorted_cov = cov.loc[sorted_tickers, sorted_tickers]

    # Step 5: Recursive bisection
    weights = recursive_bisection(sorted_cov, sorted_tickers)

    return weights


def correlation_distance(corr):
    """
    Convert correlation matrix to a proper Euclidean distance matrix.

    d(i,j) = sqrt(0.5 * (1 - ρ(i,j)))

    Properties:
      - d ∈ [0, 1]
      - d(i,i) = 0
      - d(i,j) = d(j,i)
      - Triangle inequality holds (Euclidean)

    Args:
        corr: DataFrame, correlation matrix (N × N)

    Returns:
        DataFrame, distance matrix (N × N)
    """
    # Clip correlation to [-1, 1] for numerical safety
    corr_clean = corr.clip(-1, 1)

    dist_values = np.sqrt(0.5 * (1 - corr_clean.values))

    # Force exact zero diagonal and perfect symmetry
    np.fill_diagonal(dist_values, 0.0)
    dist_values = (dist_values + dist_values.T) / 2.0

    dist = pd.DataFrame(dist_values, index=corr.index, columns=corr.columns)

    return dist


def quasi_diagonalize(link, tickers):
    """
    Reorder tickers by dendrogram leaf order so that correlated
    assets are adjacent in the covariance matrix.

    Args:
        link:    Linkage matrix from scipy hierarchical clustering.
        tickers: List of ticker strings in the original order.

    Returns:
        List of tickers in quasi-diagonal order.
    """
    order = leaves_list(link)
    return [tickers[i] for i in order]


def recursive_bisection(sorted_cov, sorted_tickers):
    """
    Allocate weights by recursively splitting the sorted ticker list
    and allocating inversely proportional to cluster variance.

    At each split:
      - Compute variance of left cluster and right cluster
      - Allocate more weight to the lower-variance cluster
      - Recurse into each sub-cluster

    Args:
        sorted_cov:     Covariance matrix reordered by quasi-diagonalization.
        sorted_tickers: Tickers in quasi-diagonal order.

    Returns:
        dict of {ticker: weight}, all positive, sum to 1.0.
    """
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]

    while clusters:
        next_clusters = []

        for cluster in clusters:
            if len(cluster) == 1:
                continue

            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            left_var = _cluster_variance(sorted_cov, left)
            right_var = _cluster_variance(sorted_cov, right)

            total_var = left_var + right_var
            if total_var == 0:
                alpha = 0.5
            else:
                # More weight to the lower-variance cluster
                alpha = 1.0 - left_var / total_var

            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)

        clusters = next_clusters

    # Normalize to sum to 1
    total = weights.sum()
    if total > 0:
        weights = weights / total

    return weights.to_dict()


def _cluster_variance(cov, tickers):
    """
    Variance of an equal-weighted portfolio of the given tickers.

    V = w' Σ w, where w = [1/n, 1/n, ..., 1/n]

    Args:
        cov:     Full covariance matrix (DataFrame).
        tickers: List of tickers in this cluster.

    Returns:
        float, portfolio variance.
    """
    sub_cov = cov.loc[tickers, tickers].values
    n = len(tickers)
    w = np.ones(n) / n
    return float(w @ sub_cov @ w)


def get_cluster_info(returns_df, linkage_method="single"):
    """
    Diagnostic: returns clustering metadata for inspection.

    Args:
        returns_df: DataFrame with dates as index, tickers as columns.

    Returns:
        dict with corr, dist, linkage, sorted_tickers, sorted_cov, weights
    """
    returns_df = returns_df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    tickers = list(returns_df.columns)

    corr = returns_df.corr()
    cov = returns_df.cov()
    cov_values = cov.values.copy()
    np.fill_diagonal(cov_values, np.diag(cov_values) + 1e-8)
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)

    dist = correlation_distance(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method=linkage_method)
    sorted_tickers = quasi_diagonalize(link, tickers)
    sorted_cov = cov.loc[sorted_tickers, sorted_tickers]
    weights = recursive_bisection(sorted_cov, sorted_tickers)

    return {
        "corr": corr,
        "dist": dist,
        "linkage": link,
        "sorted_tickers": sorted_tickers,
        "sorted_cov": sorted_cov,
        "weights": weights,
        "tickers": tickers,
    }
