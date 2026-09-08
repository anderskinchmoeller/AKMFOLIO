import numpy as np
import pandas as pd

from akm_hrp.data.hrp_universe import prepare_hrp_distance_inputs


def test_hrp_distance_inputs_are_complete_symmetric_and_psd():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-03", periods=180, freq="W-FRI")
    market = rng.normal(0.001, 0.02, size=len(dates))
    returns = pd.DataFrame(
        {
            "SPY": market + rng.normal(0.0, 0.008, size=len(dates)),
            "IWM": 1.1 * market + rng.normal(0.0, 0.012, size=len(dates)),
            "TLT": -0.2 * market + rng.normal(0.0005, 0.01, size=len(dates)),
            "GLD": rng.normal(0.0004, 0.015, size=len(dates)),
        },
        index=dates,
    )
    pit = pd.DataFrame(1, index=dates, columns=returns.columns)
    pit.iloc[:20] = 0

    prepared = prepare_hrp_distance_inputs(
        returns,
        pit_universe=pit,
        minimum_complete_weeks=104,
    )

    assert len(prepared.returns) == 160
    assert not prepared.returns.isna().any().any()
    assert np.allclose(prepared.volatility_normalized_returns.std(ddof=1), 1.0)
    assert np.allclose(prepared.angular_distance, prepared.angular_distance.T)
    assert np.allclose(np.diag(prepared.angular_distance), 0.0)
    assert np.linalg.eigvalsh(prepared.denoised_correlation).min() >= -1e-10
    assert 0.0 <= prepared.shrinkage_intensity <= 1.0


def test_volatility_scaling_does_not_change_spearman_correlation():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-05", periods=130, freq="W-FRI")
    returns = pd.DataFrame(
        rng.normal(size=(len(dates), 3)),
        index=dates,
        columns=["SPY", "TLT", "GLD"],
    )
    returns["TLT"] *= 0.1
    returns["GLD"] *= 5.0

    prepared = prepare_hrp_distance_inputs(
        returns,
        minimum_complete_weeks=104,
    )

    normalized_spearman = prepared.volatility_normalized_returns.corr(
        method="spearman"
    )
    pd.testing.assert_frame_equal(normalized_spearman, prepared.spearman_correlation)
