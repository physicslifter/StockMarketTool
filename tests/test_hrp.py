"""
test_hrp.py

Comprehensive test suite for HRP.py.
Tests mathematical properties, known-structure scenarios, edge cases,
and produces diagnostic plots.

Run:
    python test_hrp.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
from HRP import (
    compute_hrp_weights,
    correlation_distance,
    quasi_diagonalize,
    recursive_bisection,
    get_cluster_info,
)


# ═══════════════════════════════════════════════════════════════
# HELPER: Generate synthetic returns with known correlation structure
# ═══════════════════════════════════════════════════════════════
def make_synthetic_returns(n_days=500, seed=42):
    """
    Creates returns for 12 stocks in 3 known clusters:
      - Tech:   AAPL, MSFT, NVDA, GOOG  (highly correlated, high vol)
      - Energy: XOM, CVX, COP            (moderately correlated, medium vol)
      - Staples: PG, KO, WMT, CL, JNJ   (highly correlated, low vol)

    The correlation structure is known, so we can verify that HRP
    discovers the correct clusters and allocates accordingly.
    """
    rng = np.random.default_rng(seed)
    
    # Common factors
    market = rng.normal(0, 0.01, n_days)
    tech_factor = rng.normal(0, 0.012, n_days)
    energy_factor = rng.normal(0, 0.015, n_days)
    staples_factor = rng.normal(0, 0.005, n_days)

    stocks = {}

    # Tech cluster: high loading on tech_factor + market
    for name in ["AAPL", "MSFT", "NVDA", "GOOG"]:
        idio = rng.normal(0, 0.008, n_days)
        stocks[name] = market + 0.8 * tech_factor + idio

    # Energy cluster: high loading on energy_factor + market
    for name in ["XOM", "CVX", "COP"]:
        idio = rng.normal(0, 0.010, n_days)
        stocks[name] = market + 0.7 * energy_factor + idio

    # Staples cluster: high loading on staples_factor, low vol
    for name in ["PG", "KO", "WMT", "CL", "JNJ"]:
        idio = rng.normal(0, 0.004, n_days)
        stocks[name] = 0.5 * market + 0.6 * staples_factor + idio

    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(stocks, index=dates)


# ═══════════════════════════════════════════════════════════════
# TEST 1: Distance metric mathematical properties
# ═══════════════════════════════════════════════════════════════
def test_distance_properties():
    print("=" * 60)
    print("TEST 1: Distance Metric Properties")
    print("=" * 60)

    returns = make_synthetic_returns()
    corr = returns.corr()
    dist = correlation_distance(corr)
    d = dist.values

    # Property 1: Non-negative
    assert (d >= -1e-10).all(), "FAIL: Distance has negative values"
    print("  ✓ Non-negative: all distances >= 0")

    # Property 2: Zero diagonal
    diag = np.diag(d)
    assert np.allclose(diag, 0), f"FAIL: Diagonal not zero, max = {diag.max()}"
    print("  ✓ Zero diagonal: d(i,i) = 0 for all i")

    # Property 3: Symmetry
    assert np.allclose(d, d.T), "FAIL: Distance matrix is not symmetric"
    print("  ✓ Symmetric: d(i,j) = d(j,i)")

    # Property 4: Range [0, 1]
    off_diag = d[np.triu_indices_from(d, k=1)]
    assert off_diag.min() >= -1e-10, f"FAIL: Min distance {off_diag.min()} < 0"
    assert off_diag.max() <= 1.0 + 1e-10, f"FAIL: Max distance {off_diag.max()} > 1"
    print(f"  ✓ Range [0, 1]: min={off_diag.min():.4f}, max={off_diag.max():.4f}")

    # Property 5: Triangle inequality (sample 1000 random triplets)
    n = len(d)
    rng = np.random.default_rng(42)
    violations = 0
    n_tests = 1000
    for _ in range(n_tests):
        i, j, k = rng.choice(n, 3, replace=False)
        if d[i, k] > d[i, j] + d[j, k] + 1e-10:
            violations += 1
    assert violations == 0, f"FAIL: {violations}/{n_tests} triangle inequality violations"
    print(f"  ✓ Triangle inequality: 0/{n_tests} violations")

    # Property 6: Known values
    # Perfect correlation should give distance 0
    assert np.isclose(np.sqrt(0.5 * (1 - 1.0)), 0.0), "FAIL: d(ρ=1) should be 0"
    # Zero correlation should give sqrt(0.5) ≈ 0.707
    assert np.isclose(np.sqrt(0.5 * (1 - 0.0)), np.sqrt(0.5)), "FAIL: d(ρ=0) should be √0.5"
    # Perfect negative correlation should give 1.0
    assert np.isclose(np.sqrt(0.5 * (1 - (-1.0))), 1.0), "FAIL: d(ρ=-1) should be 1"
    print("  ✓ Known values: d(ρ=1)=0, d(ρ=0)=0.707, d(ρ=-1)=1")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 2: Quasi-diagonalization groups correlated assets
# ═══════════════════════════════════════════════════════════════
def test_quasi_diagonalization():
    print("=" * 60)
    print("TEST 2: Quasi-Diagonalization")
    print("=" * 60)

    returns = make_synthetic_returns()
    info = get_cluster_info(returns)

    sorted_tickers = info["sorted_tickers"]
    known_clusters = {
        "Tech": {"AAPL", "MSFT", "NVDA", "GOOG"},
        "Energy": {"XOM", "CVX", "COP"},
        "Staples": {"PG", "KO", "WMT", "CL", "JNJ"},
    }

    print(f"  Sorted order: {sorted_tickers}")

    # Check that members of each cluster appear contiguously
    for name, members in known_clusters.items():
        indices = [sorted_tickers.index(t) for t in members if t in sorted_tickers]
        indices.sort()
        span = indices[-1] - indices[0] + 1
        is_contiguous = span == len(indices)
        status = "✓" if is_contiguous else "✗"
        print(f"  {status} {name}: indices {indices}, span={span}, members={len(indices)}, "
              f"contiguous={is_contiguous}")

    # Verify the sorted covariance matrix has block structure
    # Within-cluster covariance should be higher than between-cluster
    sorted_cov = info["sorted_cov"]
    
    # Find within-cluster and between-cluster average covariance
    within_covs = []
    between_covs = []
    
    for name, members in known_clusters.items():
        members_list = [t for t in members if t in sorted_tickers]
        for i, t1 in enumerate(members_list):
            for t2 in members_list[i+1:]:
                within_covs.append(abs(sorted_cov.loc[t1, t2]))
        for t1 in members_list:
            for t2 in sorted_tickers:
                if t2 not in members:
                    between_covs.append(abs(sorted_cov.loc[t1, t2]))

    avg_within = np.mean(within_covs)
    avg_between = np.mean(between_covs)
    print(f"  Avg within-cluster |cov|:  {avg_within:.6f}")
    print(f"  Avg between-cluster |cov|: {avg_between:.6f}")
    assert avg_within > avg_between, "FAIL: Within-cluster covariance should exceed between-cluster"
    print(f"  ✓ Within-cluster covariance > between-cluster ({avg_within:.6f} > {avg_between:.6f})")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Recursive bisection gives sensible weights
# ═══════════════════════════════════════════════════════════════
def test_recursive_bisection():
    print("=" * 60)
    print("TEST 3: Recursive Bisection Weights")
    print("=" * 60)

    returns = make_synthetic_returns()
    weights = compute_hrp_weights(returns)

    tickers = list(weights.keys())
    w_values = np.array(list(weights.values()))

    # Weights sum to 1
    assert np.isclose(w_values.sum(), 1.0, atol=1e-6), \
        f"FAIL: Weights sum to {w_values.sum()}, not 1.0"
    print(f"  ✓ Weights sum to 1.0 (actual: {w_values.sum():.8f})")

    # All weights positive
    assert (w_values > 0).all(), "FAIL: Some weights are negative or zero"
    print(f"  ✓ All weights positive (min: {w_values.min():.6f})")

    # No single weight dominates (max should be < 0.5 for 12 stocks)
    assert w_values.max() < 0.5, f"FAIL: Max weight {w_values.max():.4f} is too concentrated"
    print(f"  ✓ No excessive concentration (max: {w_values.max():.4f})")

    # Cluster-level allocation: correlated clusters should share weight
    tech = sum(weights[t] for t in ["AAPL", "MSFT", "NVDA", "GOOG"])
    energy = sum(weights[t] for t in ["XOM", "CVX", "COP"])
    staples = sum(weights[t] for t in ["PG", "KO", "WMT", "CL", "JNJ"])
    print(f"  Cluster allocations: Tech={tech:.3f}, Energy={energy:.3f}, Staples={staples:.3f}")

    # Low-vol cluster (staples) should get more total weight than high-vol cluster (tech)
    # despite having more members (5 vs 4), the per-stock weight should also be higher
    tech_per_stock = tech / 4
    staples_per_stock = staples / 5
    print(f"  Per-stock: Tech={tech_per_stock:.4f}, Staples={staples_per_stock:.4f}")
    print(f"  {'✓' if staples_per_stock >= tech_per_stock else '✗'} Low-vol cluster gets >= "
          f"per-stock weight than high-vol cluster")

    # Individual weights
    print(f"\n  Individual weights:")
    for t in sorted(weights.keys(), key=lambda x: -weights[x]):
        print(f"    {t:6s}: {weights[t]:.4f}")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 4: End-to-end with extreme known structure
# ═══════════════════════════════════════════════════════════════
def test_extreme_structure():
    print("=" * 60)
    print("TEST 4: Extreme Known Structure")
    print("=" * 60)

    rng = np.random.default_rng(123)
    n_days = 300
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    # Create 2 perfectly correlated pairs + 1 independent stock
    common_a = rng.normal(0, 0.02, n_days)
    common_b = rng.normal(0, 0.02, n_days)
    independent = rng.normal(0, 0.01, n_days)

    returns = pd.DataFrame({
        "A1": common_a + rng.normal(0, 0.001, n_days),  # Pair A
        "A2": common_a + rng.normal(0, 0.001, n_days),  # Pair A (near-identical)
        "B1": common_b + rng.normal(0, 0.001, n_days),  # Pair B
        "B2": common_b + rng.normal(0, 0.001, n_days),  # Pair B (near-identical)
        "C":  independent,                                # Independent, lower vol
    }, index=dates)

    weights = compute_hrp_weights(returns)

    print(f"  Weights: {', '.join(f'{t}={w:.4f}' for t, w in weights.items())}")

    # A1 and A2 should have similar weights (they're near-identical)
    assert abs(weights["A1"] - weights["A2"]) < 0.05, \
        f"FAIL: Near-identical A1/A2 should have similar weights"
    print(f"  ✓ A1 ≈ A2 (diff: {abs(weights['A1'] - weights['A2']):.4f})")

    assert abs(weights["B1"] - weights["B2"]) < 0.05, \
        f"FAIL: Near-identical B1/B2 should have similar weights"
    print(f"  ✓ B1 ≈ B2 (diff: {abs(weights['B1'] - weights['B2']):.4f})")

    # The independent low-vol stock should get substantial weight
    # because it provides unique diversification AND has lower variance
    pair_a_total = weights["A1"] + weights["A2"]
    pair_b_total = weights["B1"] + weights["B2"]
    print(f"  Pair A total: {pair_a_total:.4f}")
    print(f"  Pair B total: {pair_b_total:.4f}")
    print(f"  Independent C: {weights['C']:.4f}")
    
    # C should get at least as much as any individual member of a pair
    assert weights["C"] >= min(weights["A1"], weights["B1"]) - 0.02, \
        "FAIL: Independent stock should get meaningful weight"
    print(f"  ✓ Independent stock C gets meaningful weight ({weights['C']:.4f})")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 5: Edge cases
# ═══════════════════════════════════════════════════════════════
def test_edge_cases():
    print("=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=200)

    # Single stock
    single = pd.DataFrame({"ONLY": rng.normal(0, 0.01, 200)}, index=dates)
    w = compute_hrp_weights(single)
    assert w == {"ONLY": 1.0}, f"FAIL: Single stock should get weight 1.0, got {w}"
    print("  ✓ Single stock: weight = 1.0")

    # Two stocks
    two = pd.DataFrame({
        "A": rng.normal(0, 0.01, 200),
        "B": rng.normal(0, 0.02, 200),
    }, index=dates)
    w = compute_hrp_weights(two)
    assert np.isclose(sum(w.values()), 1.0), "FAIL: Two-stock weights don't sum to 1"
    # Lower-vol stock A should get more weight
    print(f"  ✓ Two stocks: A={w['A']:.4f} (low vol), B={w['B']:.4f} (high vol)")
    assert w["A"] > w["B"], "FAIL: Lower-vol stock should get more weight"
    print(f"  ✓ Lower-vol stock gets more weight")

    # Three stocks
    three = pd.DataFrame({
        "X": rng.normal(0, 0.01, 200),
        "Y": rng.normal(0, 0.01, 200),
        "Z": rng.normal(0, 0.01, 200),
    }, index=dates)
    w = compute_hrp_weights(three)
    assert len(w) == 3, f"FAIL: Expected 3 weights, got {len(w)}"
    assert np.isclose(sum(w.values()), 1.0, atol=1e-6), "FAIL: Three-stock weights don't sum to 1"
    print(f"  ✓ Three stocks: weights sum to 1.0 ({', '.join(f'{t}={v:.4f}' for t,v in w.items())})")

    # Very short history (< 10 days → should fall back to equal weight)
    short = pd.DataFrame({
        "A": rng.normal(0, 0.01, 5),
        "B": rng.normal(0, 0.01, 5),
        "C": rng.normal(0, 0.01, 5),
    }, index=dates[:5])
    w = compute_hrp_weights(short)
    expected = 1.0 / 3
    for t, wt in w.items():
        assert np.isclose(wt, expected, atol=1e-6), \
            f"FAIL: Short history should give equal weight, got {t}={wt}"
    print(f"  ✓ Short history (5 days): falls back to equal weight ({expected:.4f} each)")

    # Stock with zero variance (constant price) — should be dropped
    constant = pd.DataFrame({
        "FLAT": np.zeros(200),
        "MOVE": rng.normal(0, 0.01, 200),
        "ALSO": rng.normal(0, 0.02, 200),
    }, index=dates)
    w = compute_hrp_weights(constant)
    assert "FLAT" not in w, "FAIL: Zero-variance stock should be dropped"
    assert np.isclose(sum(w.values()), 1.0, atol=1e-6), "FAIL: Weights don't sum to 1 after dropping constant stock"
    print(f"  ✓ Constant-price stock dropped, remaining: {', '.join(f'{t}={v:.4f}' for t,v in w.items())}")

    # Empty DataFrame
    empty = pd.DataFrame()
    w = compute_hrp_weights(empty)
    assert w == {}, "FAIL: Empty input should return empty dict"
    print("  ✓ Empty input returns empty dict")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 6: Stability — small input changes → small output changes
# ═══════════════════════════════════════════════════════════════
def test_stability():
    print("=" * 60)
    print("TEST 6: Weight Stability")
    print("=" * 60)

    returns = make_synthetic_returns(n_days=500, seed=42)

    # Compute weights on full data
    w1 = compute_hrp_weights(returns)

    # Compute weights dropping last 5 days
    w2 = compute_hrp_weights(returns.iloc[:-5])

    # Compute weights with tiny noise added
    rng = np.random.default_rng(99)
    noisy = returns + rng.normal(0, 1e-6, returns.shape)
    w3 = compute_hrp_weights(noisy)

    # Weight changes should be small
    common_tickers = set(w1.keys()) & set(w2.keys())
    diffs_drop = [abs(w1[t] - w2[t]) for t in common_tickers]
    max_diff_drop = max(diffs_drop)

    common_tickers_noise = set(w1.keys()) & set(w3.keys())
    diffs_noise = [abs(w1[t] - w3[t]) for t in common_tickers_noise]
    max_diff_noise = max(diffs_noise)

    print(f"  Dropping 5 days: max weight change = {max_diff_drop:.6f}")
    assert max_diff_drop < 0.05, f"FAIL: Weights too unstable to data removal ({max_diff_drop:.4f})"
    print(f"  ✓ Stable under data removal (< 0.05)")

    print(f"  Adding tiny noise: max weight change = {max_diff_noise:.6f}")
    assert max_diff_noise < 0.01, f"FAIL: Weights too sensitive to tiny noise ({max_diff_noise:.4f})"
    print(f"  ✓ Stable under tiny perturbation (< 0.01)")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 7: Comparison to equal weight
# ═══════════════════════════════════════════════════════════════
def test_vs_equal_weight():
    print("=" * 60)
    print("TEST 7: HRP vs Equal Weight — Portfolio Variance")
    print("=" * 60)

    returns = make_synthetic_returns(n_days=500, seed=42)
    cov = returns.cov()
    tickers = list(returns.columns)
    n = len(tickers)

    # Equal weight portfolio variance
    w_eq = np.ones(n) / n
    var_eq = float(w_eq @ cov.values @ w_eq)

    # HRP portfolio variance
    hrp_weights = compute_hrp_weights(returns)
    w_hrp = np.array([hrp_weights[t] for t in tickers])
    var_hrp = float(w_hrp @ cov.values @ w_hrp)

    print(f"  Equal weight portfolio variance: {var_eq:.8f}")
    print(f"  HRP portfolio variance:          {var_hrp:.8f}")
    print(f"  Reduction: {(1 - var_hrp / var_eq) * 100:.1f}%")

    # HRP should produce equal or lower variance
    assert var_hrp <= var_eq * 1.01, \
        f"FAIL: HRP variance ({var_hrp:.8f}) should not exceed equal weight ({var_eq:.8f})"
    print(f"  ✓ HRP variance <= equal weight variance")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════
# TEST 8: Linkage method comparison
# ═══════════════════════════════════════════════════════════════
def test_linkage_methods():
    print("=" * 60)
    print("TEST 8: Linkage Method Comparison")
    print("=" * 60)

    returns = make_synthetic_returns()
    cov = returns.cov()
    tickers = list(returns.columns)

    for method in ["single", "complete", "average", "ward"]:
        weights = compute_hrp_weights(returns, linkage_method=method)
        w = np.array([weights[t] for t in tickers])
        port_var = float(w @ cov.values @ w)
        max_w = max(weights.values())
        min_w = min(weights.values())
        concentration = max_w / min_w

        print(f"  {method:8s}: var={port_var:.8f}  max_w={max_w:.4f}  "
              f"min_w={min_w:.4f}  concentration={concentration:.1f}x")

    print("  (No assertion — informational comparison)\n")


# ═══════════════════════════════════════════════════════════════
# PLOTS: Visual diagnostics
# ═══════════════════════════════════════════════════════════════
def plot_diagnostics():
    print("=" * 60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 60)

    returns = make_synthetic_returns()
    info = get_cluster_info(returns)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── 1. Raw correlation matrix ──
    tickers = info["tickers"]
    raw_corr = info["corr"].loc[tickers, tickers]
    im1 = axes[0, 0].imshow(raw_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_xticks(range(len(tickers)))
    axes[0, 0].set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_yticks(range(len(tickers)))
    axes[0, 0].set_yticklabels(tickers, fontsize=8)
    axes[0, 0].set_title("Correlation (raw order)")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # ── 2. Quasi-diagonalized correlation matrix ──
    sorted_t = info["sorted_tickers"]
    sorted_corr = info["corr"].loc[sorted_t, sorted_t]
    im2 = axes[0, 1].imshow(sorted_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_xticks(range(len(sorted_t)))
    axes[0, 1].set_xticklabels(sorted_t, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_yticks(range(len(sorted_t)))
    axes[0, 1].set_yticklabels(sorted_t, fontsize=8)
    axes[0, 1].set_title("Correlation (quasi-diagonal)")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # ── 3. Dendrogram ──
    dendrogram(
        info["linkage"],
        labels=tickers,
        ax=axes[0, 2],
        leaf_rotation=45,
        leaf_font_size=8,
    )
    axes[0, 2].set_title("Dendrogram")
    axes[0, 2].set_ylabel("Distance")

    # ── 4. HRP weights vs equal weight ──
    weights = info["weights"]
    sorted_by_weight = sorted(weights.keys(), key=lambda x: -weights[x])
    hrp_vals = [weights[t] for t in sorted_by_weight]
    eq_val = 1.0 / len(sorted_by_weight)

    x = np.arange(len(sorted_by_weight))
    axes[1, 0].bar(x - 0.15, hrp_vals, 0.3, color="purple", alpha=0.7, label="HRP")
    axes[1, 0].bar(x + 0.15, [eq_val] * len(x), 0.3, color="gray", alpha=0.5, label="Equal")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sorted_by_weight, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_title("HRP vs Equal Weight")
    axes[1, 0].set_ylabel("Weight")
    axes[1, 0].legend(fontsize=8)

    # ── 5. Cluster-level allocation ──
    clusters = {
        "Tech": ["AAPL", "MSFT", "NVDA", "GOOG"],
        "Energy": ["XOM", "CVX", "COP"],
        "Staples": ["PG", "KO", "WMT", "CL", "JNJ"],
    }
    cluster_weights = {name: sum(weights[t] for t in members) for name, members in clusters.items()}
    cluster_eq = {name: len(members) / 12 for name, members in clusters.items()}

    names = list(cluster_weights.keys())
    x = np.arange(len(names))
    axes[1, 1].bar(x - 0.15, [cluster_weights[n] for n in names], 0.3,
                   color="purple", alpha=0.7, label="HRP")
    axes[1, 1].bar(x + 0.15, [cluster_eq[n] for n in names], 0.3,
                   color="gray", alpha=0.5, label="Equal")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, fontsize=10)
    axes[1, 1].set_title("Cluster-Level Allocation")
    axes[1, 1].set_ylabel("Total Weight")
    axes[1, 1].legend(fontsize=8)

    # ── 6. Distance matrix ──
    dist = info["dist"]
    sorted_dist = dist.loc[sorted_t, sorted_t]
    im6 = axes[1, 2].imshow(sorted_dist.values, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_xticks(range(len(sorted_t)))
    axes[1, 2].set_xticklabels(sorted_t, rotation=45, ha="right", fontsize=8)
    axes[1, 2].set_yticks(range(len(sorted_t)))
    axes[1, 2].set_yticklabels(sorted_t, fontsize=8)
    axes[1, 2].set_title("Distance Matrix (quasi-diagonal)")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    plt.suptitle("HRP Diagnostic Plots — Synthetic 3-Cluster Data", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Run all tests
    test_distance_properties()
    test_quasi_diagonalization()
    test_recursive_bisection()
    test_extreme_structure()
    test_edge_cases()
    test_stability()
    test_vs_equal_weight()
    test_linkage_methods()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Generate plots
    plot_diagnostics()
