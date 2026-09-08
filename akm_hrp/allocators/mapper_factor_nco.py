from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from akm_hrp.hrp.allocation import risk_contribution_regularize
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12
_WEEKS_PER_YEAR = 52.0


@dataclass(frozen=True)
class MapperFactorNCOConfig:
    """Controls the causal Mapper-conditioned factor/NCO allocator."""

    # Factor-model controls.  When no external factors are supplied, the
    # allocator learns a small statistical factor model from the asset panel.
    max_factors: int = 8
    factor_explained_variance: float = 0.80
    factor_ridge: float = 1e-4
    residual_variance_shrinkage: float = 0.25
    residual_mean_weight: float = 0.10
    expected_return_shrinkage: float = 0.70
    expected_return_winsor_quantile: float = 0.05
    annual_risk_free_rate: float = 0.0

    # Mapper constructs overlapping pullbacks of a two-dimensional lens and
    # clusters the market-state points inside each pullback with DBSCAN.
    mapper_intervals: int = 4
    mapper_overlap: float = 0.50
    mapper_min_samples: int = 3
    mapper_eps_quantile: float = 0.65
    mapper_eps_scale: float = 1.25
    mapper_max_factor_features: int = 4
    mapper_min_regime_observations: int = 26
    mapper_max_regime_fraction: float = 0.65
    mapper_regime_weight: float = 0.75
    observation_halflife: float = 52.0
    state_volatility_window: int = 13

    # Hierarchical clustering and two-level Nested Clustered Optimization.
    linkage_method: str = "average"
    max_clusters: int = 24
    optimal_leaf_ordering_max_assets: int = 100
    risk_aversion: float = 1.0
    expected_return_strength: float = 0.05
    weight_anchor_strength: float = 0.05
    turnover_penalty: float = 0.10
    optimizer_max_iterations: int = 200

    # Final portfolio controls.  The walk-forward engine supplies the separate
    # trading-cost, drift, holding-period, and rebalance-turnover constraints.
    risk_contribution_cap: float = 0.20
    min_weight: float = 0.0
    max_weight: float = 0.10


@dataclass(frozen=True)
class MapperDiagnostics:
    node_count: int
    edge_count: int
    current_node_count: int
    regime_observation_count: int
    regime_fraction: float
    effective_observation_count: float
    used_nearest_neighbour_fallback: bool
    used_regime_size_cap: bool


@dataclass(frozen=True)
class FactorModelDiagnostics:
    source: str
    factor_count: int
    explained_variance: float
    condition_number: float
    minimum_eigenvalue: float


@dataclass(frozen=True)
class NCODiagnostics:
    cluster_count: int
    cluster_sizes: tuple[int, ...]
    quasi_diagonal_order: tuple[str, ...]
    optimizer_failures: int
    expected_weekly_return: float
    expected_weekly_volatility: float
    effective_asset_count: float
    maximum_weight: float
    target_turnover_l1: float


@dataclass(frozen=True)
class MapperFactorNCODiagnostics:
    factor_model: FactorModelDiagnostics
    mapper: MapperDiagnostics
    nco: NCODiagnostics

    def as_dict(self) -> dict[str, float | int | str]:
        """Return scalar diagnostics suitable for a CSV or log record."""
        return {
            "factor_source": self.factor_model.source,
            "factor_count": self.factor_model.factor_count,
            "factor_explained_variance": self.factor_model.explained_variance,
            "covariance_condition_number": self.factor_model.condition_number,
            "covariance_minimum_eigenvalue": self.factor_model.minimum_eigenvalue,
            "mapper_node_count": self.mapper.node_count,
            "mapper_edge_count": self.mapper.edge_count,
            "mapper_current_node_count": self.mapper.current_node_count,
            "mapper_regime_observation_count": (
                self.mapper.regime_observation_count
            ),
            "mapper_regime_fraction": self.mapper.regime_fraction,
            "mapper_effective_observation_count": (
                self.mapper.effective_observation_count
            ),
            "mapper_used_nearest_neighbour_fallback": int(
                self.mapper.used_nearest_neighbour_fallback
            ),
            "mapper_used_regime_size_cap": int(
                self.mapper.used_regime_size_cap
            ),
            "nco_cluster_count": self.nco.cluster_count,
            "nco_cluster_sizes": "|".join(
                str(size) for size in self.nco.cluster_sizes
            ),
            "nco_quasi_diagonal_order": "|".join(
                str(asset) for asset in self.nco.quasi_diagonal_order
            ),
            "nco_optimizer_failures": self.nco.optimizer_failures,
            "portfolio_expected_weekly_return": self.nco.expected_weekly_return,
            "portfolio_expected_weekly_volatility": (
                self.nco.expected_weekly_volatility
            ),
            "portfolio_effective_asset_count": self.nco.effective_asset_count,
            "portfolio_maximum_weight": self.nco.maximum_weight,
            "portfolio_target_turnover_l1": self.nco.target_turnover_l1,
        }


def _normalised_exponential_weights(length: int, halflife: float) -> np.ndarray:
    """Return oldest-to-newest exponentially decaying observation weights."""
    if length < 1:
        raise ValueError("At least one observation is required.")
    if halflife <= 0.0:
        raise ValueError("observation_halflife must be positive.")
    age = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.exp2(-age / float(halflife))
    return weights / weights.sum()


def _weighted_mean_and_covariance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a normalized-weight mean and unbiased weighted covariance."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / max(float(w.sum()), _EPS)
    mean = w @ x
    centred = x - mean
    denominator = max(1.0 - float(w @ w), _EPS)
    covariance = (centred * w[:, None]).T @ centred / denominator
    covariance = np.atleast_2d(covariance)
    return mean, 0.5 * (covariance + covariance.T)


def _robust_standardise(frame: pd.DataFrame) -> pd.DataFrame:
    """Median/MAD standardization with standard-deviation fallback."""
    median = frame.median(axis=0)
    scale = 1.4826 * (frame - median).abs().median(axis=0)
    fallback = frame.std(axis=0, ddof=1)
    scale = scale.where(scale > _EPS, fallback).where(lambda x: x > _EPS, 1.0)
    return (frame - median) / scale


def _cover_intervals(
    values: np.ndarray,
    n_intervals: int,
    overlap: float,
) -> list[tuple[float, float]]:
    """Build an overlapping one-dimensional Mapper cover."""
    low = float(np.min(values))
    high = float(np.max(values))
    if n_intervals <= 1 or high - low <= _EPS:
        return [(low - _EPS, high + _EPS)]

    width = (high - low) / (n_intervals - (n_intervals - 1) * overlap)
    step = width * (1.0 - overlap)
    intervals = []
    for interval_index in range(n_intervals):
        start = low + interval_index * step
        end = start + width
        if interval_index == n_intervals - 1:
            end = max(end, high + _EPS)
        intervals.append((start - _EPS, end + _EPS))
    return intervals


def _mapper_nodes(
    state: pd.DataFrame,
    lens: np.ndarray,
    config: MapperFactorNCOConfig,
) -> list[frozenset[int]]:
    """Cluster the pullback of every non-empty two-dimensional cover cell."""
    state_values = state.to_numpy(dtype=float)
    sample_count = len(state)
    if sample_count < 2:
        return [frozenset({0})]

    neighbour_count = min(max(config.mapper_min_samples, 1) + 1, sample_count)
    neighbours = NearestNeighbors(n_neighbors=neighbour_count).fit(state_values)
    distances = neighbours.kneighbors(state_values, return_distance=True)[0]
    k_distances = distances[:, -1]
    eps = float(np.quantile(k_distances, config.mapper_eps_quantile))
    eps = max(eps * config.mapper_eps_scale, 1e-8)

    covers = [
        _cover_intervals(
            lens[:, dimension],
            config.mapper_intervals,
            config.mapper_overlap,
        )
        for dimension in range(lens.shape[1])
    ]
    nodes: list[frozenset[int]] = []
    for cell in product(*covers):
        selected = np.ones(sample_count, dtype=bool)
        for dimension, (low, high) in enumerate(cell):
            selected &= (lens[:, dimension] >= low) & (lens[:, dimension] <= high)
        positions = np.flatnonzero(selected)
        if positions.size == 0:
            continue

        if positions.size < config.mapper_min_samples:
            nodes.extend(frozenset({int(position)}) for position in positions)
            continue

        labels = DBSCAN(
            eps=eps,
            min_samples=config.mapper_min_samples,
        ).fit_predict(state_values[positions])
        for label in sorted(set(labels) - {-1}):
            nodes.append(
                frozenset(int(position) for position in positions[labels == label])
            )
        # Retain noise points as singleton nodes.  In particular, this ensures
        # the latest state always has a well-defined Mapper neighbourhood.
        nodes.extend(
            frozenset({int(position)})
            for position in positions[labels == -1]
        )

    # Overlap can produce identical pullback clusters; duplicates add no graph
    # information and would distort the node/edge diagnostics.
    return list(dict.fromkeys(nodes))


def mapper_observation_weights(
    factor_returns: pd.DataFrame,
    asset_returns: pd.DataFrame,
    config: MapperFactorNCOConfig,
) -> tuple[np.ndarray, MapperDiagnostics]:
    """Construct causal EW weights tilted to the current Mapper component."""
    if not factor_returns.index.equals(asset_returns.index):
        raise ValueError("factor_returns and asset_returns must share an index.")

    market = asset_returns.mean(axis=1)
    rolling_vol = market.rolling(
        config.state_volatility_window,
        min_periods=3,
    ).std(ddof=1)
    # Use only observations available at each historical state.  Backfilling
    # here would let an early state borrow volatility measured in later weeks.
    rolling_vol = rolling_vol.fillna(
        market.expanding(min_periods=2).std(ddof=1)
    ).fillna(0.0)
    factor_features = factor_returns.iloc[
        :, : max(1, config.mapper_max_factor_features)
    ].copy()
    factor_features.columns = [f"factor_{i}" for i in range(factor_features.shape[1])]
    state = pd.concat(
        [
            factor_features,
            market.rename("market"),
            asset_returns.std(axis=1, ddof=1).rename("dispersion"),
            asset_returns.lt(0.0).mean(axis=1).rename("downside_fraction"),
            rolling_vol.rename("rolling_market_volatility"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    state = state.fillna(state.median(axis=0)).fillna(0.0)
    state = _robust_standardise(state)

    first_lens = PCA(n_components=1, svd_solver="full").fit_transform(state)[:, 0]
    volatility_lens = state["rolling_market_volatility"].to_numpy(dtype=float)
    lens = np.column_stack([first_lens, volatility_lens])
    nodes = _mapper_nodes(state, lens, config)

    edge_count = 0
    adjacency: list[set[int]] = [set() for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if nodes[left].intersection(nodes[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
                edge_count += 1

    current_position = len(state) - 1
    current_nodes = [
        node_index
        for node_index, node in enumerate(nodes)
        if current_position in node
    ]
    regime_positions: set[int] = set()
    for node_index in current_nodes:
        regime_positions.update(nodes[node_index])
        for neighbour_index in adjacency[node_index]:
            regime_positions.update(nodes[neighbour_index])

    used_nearest_neighbour_fallback = False
    minimum_regime_size = min(
        max(config.mapper_min_regime_observations, 1),
        len(state),
    )
    if len(regime_positions) < minimum_regime_size:
        used_nearest_neighbour_fallback = True
        current_state = state.to_numpy(dtype=float)[-1]
        distances = np.linalg.norm(state.to_numpy(dtype=float) - current_state, axis=1)
        nearest = np.argsort(distances)[:minimum_regime_size]
        regime_positions.update(int(position) for position in nearest)

    # Overlapping cover cells and one-hop neighbours can occasionally connect
    # most of the window.  Keep the topology as the candidate set, then retain
    # the candidates closest to today's standardized state so the conditioning
    # remains meaningfully local.
    maximum_regime_size = min(
        len(state),
        max(
            minimum_regime_size,
            int(np.floor(config.mapper_max_regime_fraction * len(state))),
        ),
    )
    used_regime_size_cap = len(regime_positions) > maximum_regime_size
    if used_regime_size_cap:
        state_values = state.to_numpy(dtype=float)
        distances = np.linalg.norm(state_values - state_values[-1], axis=1)
        candidates = np.array(sorted(regime_positions), dtype=int)
        selected = candidates[np.argsort(distances[candidates])[:maximum_regime_size]]
        regime_positions = set(int(position) for position in selected)

    regime_indicator = np.zeros(len(state), dtype=float)
    regime_indicator[list(regime_positions)] = 1.0
    regime_weight = float(np.clip(config.mapper_regime_weight, 0.0, 1.0))
    topology_multiplier = (1.0 - regime_weight) + regime_weight * regime_indicator
    weights = _normalised_exponential_weights(
        len(state),
        config.observation_halflife,
    )
    weights *= topology_multiplier
    weights /= weights.sum()

    diagnostics = MapperDiagnostics(
        node_count=len(nodes),
        edge_count=edge_count,
        current_node_count=len(current_nodes),
        regime_observation_count=len(regime_positions),
        regime_fraction=float(len(regime_positions) / len(state)),
        effective_observation_count=float(1.0 / np.sum(weights**2)),
        used_nearest_neighbour_fallback=used_nearest_neighbour_fallback,
        used_regime_size_cap=used_regime_size_cap,
    )
    return weights, diagnostics


def _statistical_factors(
    returns: pd.DataFrame,
    config: MapperFactorNCOConfig,
) -> tuple[pd.DataFrame, float]:
    """Extract deterministic low-rank statistical factors with randomized PCA."""
    maximum = min(config.max_factors, returns.shape[0] - 1, returns.shape[1])
    if maximum < 1:
        raise ValueError("Too few observations or assets for a factor model.")
    pca = PCA(
        n_components=maximum,
        svd_solver="randomized" if maximum < min(returns.shape) else "full",
        random_state=0,
    ).fit(returns.to_numpy(dtype=float))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    factor_count = int(
        np.searchsorted(cumulative, config.factor_explained_variance, side="left") + 1
    )
    factor_count = min(maximum, max(1, factor_count))
    loadings = pca.components_[:factor_count].T

    # Project uncentered returns.  Unlike PCA.transform, this preserves each
    # factor's risk premium for the expected-return model while covariance
    # calculations below still centre the factor series explicitly.
    values = returns.to_numpy(dtype=float) @ loadings
    factors = pd.DataFrame(
        values,
        index=returns.index,
        columns=[f"PCA_{i + 1}" for i in range(factor_count)],
    )
    return factors, float(cumulative[factor_count - 1])


def _nearest_positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.median(np.diag(covariance))), _EPS)
    eigenvalues = np.clip(eigenvalues, scale * 1e-8, None)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def fit_mapper_factor_model(
    asset_returns: pd.DataFrame,
    config: MapperFactorNCOConfig,
    factor_returns: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame, FactorModelDiagnostics, MapperDiagnostics]:
    """Estimate regime-conditioned expected returns and factor covariance."""
    returns = asset_returns.copy().astype(float).sort_index()
    if factor_returns is None:
        factors, explained_variance = _statistical_factors(returns, config)
        source = "statistical_pca"
    else:
        aligned = (
            factor_returns.copy()
            .astype(float)
            .sort_index()
            .reindex(returns.index)
            .replace([np.inf, -np.inf], np.nan)
        )
        valid_rows = aligned.notna().all(axis=1)
        returns = returns.loc[valid_rows]
        factors = aligned.loc[valid_rows]
        nonconstant = factors.std(axis=0, ddof=1) > _EPS
        factors = factors.loc[:, nonconstant]
        if len(returns) < max(config.mapper_min_regime_observations, 3):
            raise ValueError(
                "External factors have too few complete rows aligned to returns."
            )
        if factors.shape[1] < 1:
            raise ValueError("External factor returns contain no varying columns.")
        if factors.shape[1] > config.max_factors:
            factors = factors.iloc[:, : config.max_factors]
        explained_variance = float("nan")
        source = "external"

    observation_weights, mapper_diagnostics = mapper_observation_weights(
        factors,
        returns,
        config,
    )
    x = returns.to_numpy(dtype=float)
    f = factors.to_numpy(dtype=float)
    asset_mean, _ = _weighted_mean_and_covariance(x, observation_weights)
    factor_mean, factor_covariance = _weighted_mean_and_covariance(
        f,
        observation_weights,
    )
    x_centered = x - asset_mean
    f_centered = f - factor_mean

    weighted_factors = f_centered * np.sqrt(observation_weights[:, None])
    weighted_assets = x_centered * np.sqrt(observation_weights[:, None])
    gram = weighted_factors.T @ weighted_factors
    ridge_scale = max(float(np.trace(gram) / max(len(gram), 1)), _EPS)
    regularized_gram = gram + config.factor_ridge * ridge_scale * np.eye(len(gram))
    beta = np.linalg.solve(
        regularized_gram,
        weighted_factors.T @ weighted_assets,
    )

    residuals = x_centered - f_centered @ beta
    _, residual_covariance = _weighted_mean_and_covariance(
        residuals,
        observation_weights,
    )
    residual_variance = np.clip(np.diag(residual_covariance), _EPS, None)
    residual_centre = float(np.median(residual_variance))
    residual_shrinkage = float(
        np.clip(config.residual_variance_shrinkage, 0.0, 1.0)
    )
    residual_variance = (
        (1.0 - residual_shrinkage) * residual_variance
        + residual_shrinkage * residual_centre
    )
    covariance_values = (
        beta.T @ factor_covariance @ beta + np.diag(residual_variance)
    )
    covariance_values = _nearest_positive_semidefinite(covariance_values)

    residual_mean = asset_mean - factor_mean @ beta
    expected = (
        factor_mean @ beta
        + float(np.clip(config.residual_mean_weight, 0.0, 1.0)) * residual_mean
        - config.annual_risk_free_rate / _WEEKS_PER_YEAR
    )
    centre = float(np.median(expected))
    shrinkage = float(np.clip(config.expected_return_shrinkage, 0.0, 1.0))
    expected = (1.0 - shrinkage) * expected + shrinkage * centre
    winsor = float(np.clip(config.expected_return_winsor_quantile, 0.0, 0.49))
    if winsor > 0.0 and len(expected) >= 4:
        lower, upper = np.quantile(expected, [winsor, 1.0 - winsor])
        expected = np.clip(expected, lower, upper)

    covariance = pd.DataFrame(
        covariance_values,
        index=returns.columns,
        columns=returns.columns,
    )
    expected_returns = pd.Series(expected, index=returns.columns, dtype=float)
    eigenvalues = np.linalg.eigvalsh(covariance_values)
    diagnostics = FactorModelDiagnostics(
        source=source,
        factor_count=factors.shape[1],
        explained_variance=explained_variance,
        condition_number=float(np.linalg.cond(covariance_values)),
        minimum_eigenvalue=float(eigenvalues[0]),
    )
    return expected_returns, covariance, diagnostics, mapper_diagnostics


def _covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    values = covariance.to_numpy(dtype=float)
    volatility = np.sqrt(np.clip(np.diag(values), _EPS, None))
    correlation = values / np.outer(volatility, volatility)
    correlation = np.clip(0.5 * (correlation + correlation.T), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return pd.DataFrame(correlation, index=covariance.index, columns=covariance.columns)


def hierarchical_quasi_diagonal_clusters(
    covariance: pd.DataFrame,
    config: MapperFactorNCOConfig,
) -> tuple[list[list[str]], tuple[str, ...]]:
    """Create ordered asset clusters and the hierarchy's leaf ordering."""
    if covariance.shape[0] < 2:
        raise ValueError("Hierarchical clustering requires at least two assets.")
    correlation = _covariance_to_correlation(covariance)
    distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation.to_numpy()), 0.0))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(
        squareform(distance, checks=False),
        method=config.linkage_method,
        # Exact optimal leaf ordering becomes expensive for institutional-size
        # universes.  Standard linkage leaves are still a valid hierarchical
        # quasi-diagonalization and leave the cluster partition unchanged.
        optimal_ordering=(
            len(covariance) <= config.optimal_leaf_ordering_max_assets
        ),
    )
    assets = list(covariance.index)
    order_positions = leaves_list(tree).astype(int)
    order = tuple(assets[position] for position in order_positions)

    cluster_count = min(
        max(2, int(round(np.sqrt(len(assets))))),
        max(2, config.max_clusters),
        len(assets),
    )
    # A distance-threshold tree cut can be pathologically unbalanced (for
    # example, one 461-asset cluster plus many singletons in a 500-name
    # universe), which defeats NCO's dimensionality reduction.  Contiguous
    # blocks of the quasi-diagonal leaf order remain correlation-local while
    # guaranteeing that every intra-cluster optimization is genuinely small.
    clusters = [
        list(block)
        for block in np.array_split(np.asarray(order, dtype=object), cluster_count)
        if len(block) > 0
    ]
    return clusters, order


def _utility_solution(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    config: MapperFactorNCOConfig,
    previous_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Solve one long-only, fully invested utility problem inside NCO."""
    asset_count = len(expected_returns)
    if asset_count == 1:
        return np.ones(1, dtype=float), True
    variance = np.clip(np.diag(covariance), _EPS, None)
    inverse_volatility = 1.0 / np.sqrt(variance)
    prior = inverse_volatility / inverse_volatility.sum()

    covariance_scale = max(float(np.median(variance)), _EPS)
    normalized_covariance = covariance / covariance_scale
    expected_centre = float(np.median(expected_returns))
    expected_scale = 1.4826 * float(
        np.median(np.abs(expected_returns - expected_centre))
    )
    if expected_scale <= _EPS:
        expected_scale = max(float(np.std(expected_returns, ddof=0)), _EPS)
    expected_score = (expected_returns - expected_centre) / expected_scale

    previous = None
    if previous_weights is not None:
        previous = np.clip(np.asarray(previous_weights, dtype=float), 0.0, None)
        if float(previous.sum()) > _EPS:
            previous /= previous.sum()
        else:
            previous = None

    def objective(weights: np.ndarray) -> float:
        risk = 0.5 * config.risk_aversion * float(
            weights @ normalized_covariance @ weights
        )
        reward = config.expected_return_strength * float(expected_score @ weights)
        anchor = config.weight_anchor_strength * float(np.sum((weights - prior) ** 2))
        turnover = 0.0
        if previous is not None:
            difference = weights - previous
            # A quadratic tracking penalty is deliberately used inside NCO:
            # it is well-conditioned and convex.  The walk-forward engine
            # separately enforces the actual hard L1 rebalance-turnover cap.
            turnover = config.turnover_penalty * float(difference @ difference)
        return risk - reward + anchor + turnover

    def gradient(weights: np.ndarray) -> np.ndarray:
        derivative = (
            config.risk_aversion * (normalized_covariance @ weights)
            - config.expected_return_strength * expected_score
            + 2.0 * config.weight_anchor_strength * (weights - prior)
        )
        if previous is not None:
            difference = weights - previous
            derivative = derivative + 2.0 * config.turnover_penalty * difference
        return derivative

    result = minimize(
        objective,
        prior,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, 1.0)] * asset_count,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
                "jac": lambda weights: np.ones_like(weights),
            }
        ],
        options={
            "maxiter": config.optimizer_max_iterations,
            "ftol": 1e-10,
            "disp": False,
        },
    )
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or float(result.x.sum()) <= _EPS
    ):
        return prior, False
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    return weights / weights.sum(), True


def nested_clustered_optimization(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: MapperFactorNCOConfig,
    previous_weights: pd.Series | None = None,
) -> tuple[pd.Series, NCODiagnostics]:
    """Solve intra-cluster portfolios, then optimize across cluster portfolios."""
    assets = list(covariance.index)
    expected_returns = expected_returns.reindex(assets)
    clusters, order = hierarchical_quasi_diagonal_clusters(covariance, config)
    basis = np.zeros((len(assets), len(clusters)), dtype=float)
    failures = 0

    aligned_previous = None
    if previous_weights is not None:
        aligned_previous = previous_weights.reindex(assets).fillna(0.0)

    for cluster_index, cluster in enumerate(clusters):
        positions = np.array([assets.index(asset) for asset in cluster], dtype=int)
        previous_inside = None
        if aligned_previous is not None:
            previous_inside = aligned_previous.iloc[positions].to_numpy(dtype=float)
        solution, success = _utility_solution(
            expected_returns.iloc[positions].to_numpy(dtype=float),
            covariance.iloc[positions, positions].to_numpy(dtype=float),
            config,
            previous_inside,
        )
        failures += int(not success)
        basis[positions, cluster_index] = solution

    expected_cluster_returns = basis.T @ expected_returns.to_numpy(dtype=float)
    cluster_covariance = basis.T @ covariance.to_numpy(dtype=float) @ basis
    previous_clusters = None
    if aligned_previous is not None:
        previous_clusters = np.array(
            [float(aligned_previous.reindex(cluster).fillna(0.0).sum()) for cluster in clusters]
        )
    cluster_weights, success = _utility_solution(
        expected_cluster_returns,
        cluster_covariance,
        config,
        previous_clusters,
    )
    failures += int(not success)
    raw_values = basis @ cluster_weights
    raw = pd.Series(raw_values / raw_values.sum(), index=assets, dtype=float)

    risk_regularized = risk_contribution_regularize(
        raw,
        covariance,
        cap=config.risk_contribution_cap,
    )
    final = apply_bounds(
        risk_regularized,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
    )

    covariance_values = covariance.to_numpy(dtype=float)
    final_values = final.to_numpy(dtype=float)
    expected_return = float(expected_returns.to_numpy(dtype=float) @ final_values)
    expected_volatility = float(
        np.sqrt(max(final_values @ covariance_values @ final_values, 0.0))
    )
    turnover = 0.0
    if aligned_previous is not None:
        previous = aligned_previous.clip(lower=0.0)
        if float(previous.sum()) > _EPS:
            previous /= previous.sum()
            turnover = float((final - previous).abs().sum())
    diagnostics = NCODiagnostics(
        cluster_count=len(clusters),
        cluster_sizes=tuple(len(cluster) for cluster in clusters),
        quasi_diagonal_order=order,
        optimizer_failures=failures,
        expected_weekly_return=expected_return,
        expected_weekly_volatility=expected_volatility,
        effective_asset_count=float(1.0 / np.sum(final_values**2)),
        maximum_weight=float(final.max()),
        target_turnover_l1=turnover,
    )
    return final, diagnostics


class MapperFactorNCOAllocator:
    """Factor expected returns + Mapper regimes + hierarchical NCO weights."""

    def __init__(
        self,
        config: MapperFactorNCOConfig | None = None,
        factor_returns: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or MapperFactorNCOConfig()
        self.factor_returns = factor_returns
        self.last_diagnostics: MapperFactorNCODiagnostics | None = None
        self.last_expected_returns: pd.Series | None = None
        self.last_factor_covariance: pd.DataFrame | None = None
        self._previous_weights: pd.Series | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.max_factors < 1:
            raise ValueError("max_factors must be at least one.")
        if not 0.0 < self.config.factor_explained_variance <= 1.0:
            raise ValueError("factor_explained_variance must be in (0, 1].")
        if self.config.mapper_intervals < 1:
            raise ValueError("mapper_intervals must be at least one.")
        if self.config.mapper_min_samples < 1:
            raise ValueError("mapper_min_samples must be at least one.")
        if not 0.0 <= self.config.mapper_overlap < 1.0:
            raise ValueError("mapper_overlap must be in [0, 1).")
        if not 0.0 <= self.config.mapper_eps_quantile <= 1.0:
            raise ValueError("mapper_eps_quantile must be in [0, 1].")
        if not 0.0 < self.config.mapper_max_regime_fraction <= 1.0:
            raise ValueError("mapper_max_regime_fraction must be in (0, 1].")
        if self.config.mapper_eps_scale <= 0.0:
            raise ValueError("mapper_eps_scale must be positive.")
        if self.config.factor_ridge < 0.0:
            raise ValueError("factor_ridge cannot be negative.")
        if not 0.0 < self.config.risk_contribution_cap <= 1.0:
            raise ValueError("risk_contribution_cap must be in (0, 1].")
        if self.config.min_weight < 0.0:
            raise ValueError("min_weight cannot be negative.")
        if self.config.max_weight <= 0.0:
            raise ValueError("max_weight must be positive.")

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self.last_expected_returns = None
        self.last_factor_covariance = None
        self._previous_weights = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        window = (
            returns.copy()
            .astype(float)
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
        )
        if window.index.has_duplicates:
            raise ValueError("returns index contains duplicate dates.")
        valid = (
            (window.notna().sum(axis=0) >= 3)
            & window.std(axis=0, skipna=True).notna()
            & (window.std(axis=0, skipna=True) > _EPS)
        )
        window = window.loc[:, valid]
        if window.shape[1] < 2:
            raise ValueError("Too few non-constant assets remain for Mapper Factor NCO.")
        window = window.fillna(window.mean(axis=0))

        expected_returns, covariance, factor_diagnostics, mapper_diagnostics = (
            fit_mapper_factor_model(
                window,
                self.config,
                factor_returns=self.factor_returns,
            )
        )
        weights, nco_diagnostics = nested_clustered_optimization(
            expected_returns,
            covariance,
            self.config,
            previous_weights=self._previous_weights,
        )
        self._previous_weights = weights.copy()
        self.last_expected_returns = expected_returns.copy()
        self.last_factor_covariance = covariance.copy()
        self.last_diagnostics = MapperFactorNCODiagnostics(
            factor_model=factor_diagnostics,
            mapper=mapper_diagnostics,
            nco=nco_diagnostics,
        )
        return weights
