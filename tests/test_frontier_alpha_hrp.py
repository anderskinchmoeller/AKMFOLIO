import numpy as np
import pandas as pd

from adaptive_consensus_hrp_carry import (
    ModelConfig,
    _turnover_limited_target,
    run_backtest,
)
from frontier_alpha_hrp import (
    CostAwareMultiHorizonSignalEnsemble,
    _risk_metrics,
    _save_plots,
)
from hrp_alpha_validation import (
    ValidationConfig,
    _reality_check,
    _stationary_bootstrap_indices,
    _superior_predictive_ability_test,
    evaluate,
)


class _CountingAllocator:
    def __init__(self):
        self.calls = 0
        self.last_core = None
        self.last_legacy = None
        self.last_price_only = None
        self.last_diagnostics = None

    def allocate(self, returns, fundamental_snapshot=None):
        self.calls += 1
        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        self.last_core = weights
        self.last_legacy = weights
        self.last_price_only = weights
        return weights

    def observe(self, realized_returns):
        return None


def test_expert_turnover_includes_assets_that_leave_the_universe():
    config = ModelConfig(transaction_cost_bps=10.0)
    ensemble = CostAwareMultiHorizonSignalEnsemble(config)
    ensemble.positions = pd.DataFrame(
        {"signal__h01": [1.0, 0.0, -1.0]},
        index=["A", "B", "C"],
    )
    current = pd.DataFrame(
        {"signal__h01": [1.0, 0.0, -1.0]},
        index=["B", "C", "D"],
    )

    ensemble._set_pending_costs(current)

    assert np.isclose(ensemble.last_expert_turnover["signal__h01"], 2.0)
    assert np.isclose(ensemble._pending_costs["signal__h01"], 0.002)


def test_initial_portfolio_and_expert_deployment_are_charged():
    config = ModelConfig(transaction_cost_bps=10.0)
    target = pd.Series({"A": 0.6, "B": 0.4})
    deployed, turnover = _turnover_limited_target(
        pd.Series(0.0, index=target.index),
        target,
        config,
    )
    ensemble = CostAwareMultiHorizonSignalEnsemble(config)
    ensemble._set_pending_costs(
        pd.DataFrame({"signal__h01": [1.0, -1.0]}, index=target.index)
    )

    assert deployed.equals(target)
    assert np.isclose(turnover, 1.0)
    assert np.isclose(ensemble.last_expert_turnover["signal__h01"], 1.0)
    assert np.isclose(ensemble._pending_costs["signal__h01"], 0.001)


def test_forced_exit_uses_maintenance_trade_between_scheduled_allocations():
    dates = pd.date_range("2020-01-03", periods=9, freq="W-FRI")
    returns = pd.DataFrame(
        {
            "A": np.linspace(-0.01, 0.01, len(dates)),
            "B": np.linspace(0.02, -0.005, len(dates)),
            "C": np.tile([0.005, -0.004, 0.002], 3),
        },
        index=dates,
    )
    pit = pd.DataFrame(True, index=dates, columns=returns.columns)
    pit.loc[dates[2]:, "A"] = False
    allocator = _CountingAllocator()
    config = ModelConfig(
        lookback_weeks=4,
        minimum_observations=2,
        rebalance_weeks=4,
        max_weight=1.0,
    )

    _, turnover, weights, _, _ = run_backtest(
        returns,
        pit,
        config,
        evaluation_start=None,
        missing_held_return="zero",
        allocator_override=allocator,
        integrated_strategy_name="test_frontier",
        maintenance_only_forced_exits=True,
    )

    assert allocator.calls == 2
    assert weights.loc[dates[3], "A"] == 0.0
    assert turnover.loc[dates[3], "test_frontier"] > 0.0


def test_voluntary_zero_weight_does_not_bypass_turnover_budget():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.0, "B": 1.0})
    eligible = pd.Series({"A": True, "B": True})
    config = ModelConfig(
        no_trade_l1=0.0,
        maximum_rebalance_turnover_l1=0.2,
    )

    limited, traded = _turnover_limited_target(
        current,
        target,
        config,
        mandatory_eligible=eligible,
    )

    assert np.isclose(traded, 0.2)
    assert np.allclose(limited.to_numpy(), [0.4, 0.6])


def test_expert_reward_is_net_of_its_own_pending_trading_cost():
    ensemble = CostAwareMultiHorizonSignalEnsemble(ModelConfig())
    ensemble.positions = pd.DataFrame(
        {"signal__h01": [1.0, -1.0]},
        index=["A", "B"],
    )
    ensemble._pending_costs = pd.Series({"signal__h01": 0.003})

    ensemble.observe(pd.Series([0.0, 0.0], index=["A", "B"], name="2024-01-05"))

    assert np.isclose(ensemble.payoff_history.iloc[-1, 0], -0.003)
    assert np.isclose(ensemble._pending_costs["signal__h01"], 0.0)


def test_multi_horizon_floor_does_not_force_equal_expert_weights():
    ensemble = CostAwareMultiHorizonSignalEnsemble(ModelConfig())
    names = pd.Index(["momentum_12_1__h01", "low_volatility__h01"])
    ensemble._expert_base_names = {
        "momentum_12_1__h01": "momentum_12_1",
        "low_volatility__h01": "low_volatility",
    }
    dates = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    ensemble.payoff_history = pd.DataFrame(
        {
            "momentum_12_1__h01": np.full(len(dates), 0.005),
            "low_volatility__h01": np.full(len(dates), -0.005),
        },
        index=dates,
    )

    weights, _ = ensemble._weights(names)

    assert weights["momentum_12_1__h01"] > weights["low_volatility__h01"]
    assert weights.nunique() == 2


def test_expert_maintenance_exits_deleted_assets_and_charges_cost():
    ensemble = CostAwareMultiHorizonSignalEnsemble(
        ModelConfig(transaction_cost_bps=10.0)
    )
    ensemble.positions = pd.DataFrame(
        {"signal__h01": [1.0, 0.25, -1.25]},
        index=["A", "B", "C"],
    )
    eligible = pd.Series({"A": False, "B": True, "C": True})

    ensemble.maintain_universe(eligible)

    assert "A" not in ensemble.positions.index
    assert ensemble._pending_costs["signal__h01"] > 0.0
    assert np.isclose(ensemble.positions["signal__h01"].abs().sum(), 1.0)
    assert np.isclose(ensemble.positions["signal__h01"].sum(), 0.0)


def test_spa_reports_selected_model_familywise_adjusted_probability():
    rng = np.random.default_rng(42)
    observations = 800
    dates = pd.date_range("2000-01-07", periods=observations, freq="W-FRI")
    benchmark = pd.Series(
        rng.normal(0.0005, 0.018, observations),
        index=dates,
        name="benchmark",
    )
    candidates = pd.DataFrame(
        {
            "strong": benchmark + rng.normal(0.0020, 0.004, observations),
            "null": benchmark + rng.normal(0.0, 0.004, observations),
            "inferior": benchmark + rng.normal(-0.0010, 0.004, observations),
        },
        index=dates,
    )
    indices = _stationary_bootstrap_indices(
        observations,
        samples=1_000,
        mean_block_length=10.0,
        rng=np.random.default_rng(7),
    )

    spa = _superior_predictive_ability_test(
        candidates,
        benchmark,
        indices,
        selected_candidate="strong",
    )
    reality = _reality_check(
        candidates,
        benchmark,
        indices,
        selected_candidate="strong",
    )

    assert spa["selected_candidate"] == "strong"
    assert spa["selected_adjusted_p_value"] <= 0.05
    assert reality["selected_adjusted_p_value"] <= 0.05
    assert set(spa["adjusted_p_values"]) == set(candidates.columns)


def test_trial_correlation_is_estimated_from_active_not_total_returns():
    rng = np.random.default_rng(11)
    observations = 320
    dates = pd.date_range("2000-01-07", periods=observations, freq="W-FRI")
    benchmark = pd.Series(rng.normal(0.001, 0.04, observations), index=dates)
    candidate_a = benchmark + rng.normal(0.0002, 0.002, observations)
    candidate_b = benchmark + rng.normal(0.0002, 0.002, observations)
    returns = pd.DataFrame(
        {"benchmark": benchmark, "candidate_a": candidate_a}, index=dates
    )
    candidates = pd.DataFrame(
        {"candidate_a": candidate_a, "candidate_b": candidate_b}, index=dates
    )
    config = ValidationConfig(
        research_trials=20,
        bootstrap_samples=100,
        maximum_cscv_paths=50,
    )

    summary, *_ = evaluate(
        returns,
        "candidate_a",
        "benchmark",
        candidates,
        config,
    )

    assert summary["mean_trial_correlation_source"] == (
        "estimated_from_registered_active_returns"
    )
    assert summary["estimated_candidate_correlation"] < 0.2
    assert summary["effective_trials"] > 15.0


def test_risk_metrics_include_tail_active_and_turnover_statistics():
    dates = pd.date_range("2020-01-03", periods=104, freq="W-FRI")
    benchmark = pd.Series(np.tile([0.01, -0.005], 52), index=dates)
    strategy = benchmark + 0.0005
    returns = pd.DataFrame({"strategy": strategy, "benchmark": benchmark})
    turnover = pd.DataFrame(0.02, index=dates, columns=returns.columns)

    metrics = _risk_metrics(returns, turnover, "benchmark")

    expected = {
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "weekly_var_95",
        "weekly_cvar_95",
        "annual_turnover_l1",
        "tracking_error",
        "information_ratio",
    }
    assert expected.issubset(metrics.columns)
    assert metrics.loc["strategy", "annual_active_return"] > 0.0


def test_frontier_plots_include_current_weight_png(tmp_path):
    dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    returns = pd.DataFrame(
        {
            "frontier_alpha_hrp": np.linspace(-0.01, 0.015, len(dates)),
            "robust_consensus_core": np.linspace(-0.008, 0.012, len(dates)),
        },
        index=dates,
    )
    current = pd.DataFrame(
        {
            "date": ["2025-12-26"] * 3,
            "model": ["frontier_alpha_hrp"] * 3,
            "asset": ["AAPL", "MSFT", "NVDA"],
            "permno": ["14593", "10107", "86580"],
            "weight": [0.40, 0.35, 0.25],
        }
    )

    _save_plots(returns, current, tmp_path)

    for name in (
        "equity_curves.png",
        "drawdowns.png",
        "rolling_sharpe.png",
        "current_weights.png",
    ):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
