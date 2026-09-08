import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from akm_hrp.allocators.dynamic_barra_alpha import (
    DynamicBarraAlphaAllocator,
    DynamicBarraAlphaConfig,
    CovarianceBackend,
    _factor_risk_model,
)
from akm_hrp.allocators.retail_edge_mpc import RetailEdgeMPCAllocator, RetailEdgeMPCConfig
# ---------- 1. Rotation-invariant spectral NN ----------

class RINNSpectralNet(nn.Module):
    """
    Simple 1D spectral network: maps eigenvalues -> shrunk eigenvalues.
    Rotation-invariant because we keep eigenvectors fixed.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        # eigenvalues: shape (n, 1), sorted
        return self.net(eigenvalues)


class RINNCovarianceBackend(CovarianceBackend):
    """
    Covariance backend that:
    - builds baseline factor risk covariance via _factor_risk_model,
    - applies a learned spectral shrinkage to eigenvalues.
    """

    def __init__(self, model: RINNSpectralNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def estimate(
        self,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        sectors: pd.Series,
        market_caps: pd.Series,
        config: DynamicBarraAlphaConfig,
    ) -> tuple[pd.DataFrame, pd.Series]:
        # Baseline factor risk model
        covariance, specific_volatility = _factor_risk_model(
            returns,
            exposures,
            sectors,
            market_caps,
            config,
        )
        Sigma = covariance.to_numpy(dtype=float)
        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(Sigma)
        # Sort eigenvalues for rotation-invariant NN
        order = np.argsort(vals)
        vals_sorted = vals[order]
        ev = torch.tensor(vals_sorted[:, None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            shrunk_sorted = self.model(ev).squeeze().cpu().numpy()
        # Restore original ordering
        inv_order = np.argsort(order)
        shrunk = shrunk_sorted[inv_order]
        # Rebuild covariance
        Sigma_shrunk = (vecs * shrunk) @ vecs.T
        Sigma_shrunk = 0.5 * (Sigma_shrunk + Sigma_shrunk.T)
        cov_df = pd.DataFrame(
            Sigma_shrunk, index=covariance.index, columns=covariance.columns
        )
        return cov_df, specific_volatility


# ---------- 2. Load your data ----------

# Replace these with your actual data sources

balanced_pit = pd.read_csv("wrds_top2500/pit_universe.csv", index_col=0)
balanced_pit.index = pd.to_datetime(balanced_pit.index)
balanced_pit = balanced_pit.astype(bool)
returns = pd.read_csv("wrds_top2500/balanced_hrp/weekly_returns.csv", index_col=0)
returns.index = pd.to_datetime(returns.index)
returns = returns.sort_index().astype(float)

features = pd.read_csv("wrds_full_clean/structural_alpha_features.csv.gz")

sectors_history = pd.read_csv("wrds_full_clean/crsp_sector_history.csv.gz")
# Ensure indices and columns are aligned as in DynamicBarraAlphaAllocator expectations
returns = returns.sort_index().astype(float)


# ---------- 3. Instantiate RI-NN model and backend ----------

rin_model = RINNSpectralNet(hidden_dim=64)

# TODO: load trained weights here, e.g.:
# rin_model.load_state_dict(torch.load("rin_covariance_model.pt"))

rin_backend = RINNCovarianceBackend(rin_model, device="cpu")


# ---------- 4. Instantiate DynamicBarraAlphaAllocator with RI-NN backend ----------

barra_config = DynamicBarraAlphaConfig(
    portfolio_value=1_000_000.0,
    weekly_cvar_95_limit=0.06,
)

barra_allocator = DynamicBarraAlphaAllocator(
    balanced_pit=balanced_pit,
    structural_features=features,
    sector_history=sectors_history,
    config=barra_config,
    covariance_backend=rin_backend,   # <-- plug-in RI-NN backend
)


# ---------- 5. Instantiate RetailEdgeMPCAllocator (optional overlay) ----------

edge_config = RetailEdgeMPCConfig(
    portfolio_value=250_000.0,
    weekly_cvar_95_limit=0.12,
)

edge_allocator = RetailEdgeMPCAllocator(
    balanced_pit=balanced_pit,
    config=edge_config,
)


# ---------- 6. Walk-forward backtest ----------

weights_barra = {}
weights_edge = {}
portfolio_returns_barra = []
portfolio_returns_edge = []
dates = []

# Warmup: ensure enough history for risk model
start_idx = 200  # e.g. 200 weeks of history before first allocation

for i in range(start_idx, len(returns.index) - 1):
    as_of = returns.index[i]
    next_date = returns.index[i + 1]

    # Use all history up to as_of
    window = returns.loc[:as_of]

    # --- Barra core with RI-NN covariance ---
    w_barra = barra_allocator.allocate(window)
    weights_barra[as_of] = w_barra

    # Realized return next period
    r_next = returns.loc[next_date, w_barra.index].fillna(0.0)
    pnl_barra = float((w_barra * r_next).sum())
    portfolio_returns_barra.append(pnl_barra)

    # --- Retail edge MPC overlay (optional) ---
    w_edge = edge_allocator.allocate(window)
    weights_edge[as_of] = w_edge

    r_next_edge = returns.loc[next_date, w_edge.index].fillna(0.0)
    pnl_edge = float((w_edge * r_next_edge).sum())
    portfolio_returns_edge.append(pnl_edge)

    dates.append(next_date)

# Convert results to DataFrames/Series
barra_weights_df = pd.DataFrame(weights_barra).T
edge_weights_df = pd.DataFrame(weights_edge).T
barra_pnl = pd.Series(portfolio_returns_barra, index=dates, name="barra_pnl")
edge_pnl = pd.Series(portfolio_returns_edge, index=dates, name="edge_pnl")

# ---------- 7. Simple performance summary ----------

def summarize_pnl(pnl: pd.Series, label: str) -> None:
    mean = float(pnl.mean())
    vol = float(pnl.std(ddof=1))
    sharpe = mean / vol if vol > 0 else 0.0
    print(f"{label}: mean={mean:.4%}, vol={vol:.4%}, sharpe={sharpe:.2f}")

summarize_pnl(barra_pnl, "Barra + RI-NN covariance")
summarize_pnl(edge_pnl, "Retail Edge MPC")


# ---------- 8. Access diagnostics if you want ----------

last_diag_barra = barra_allocator.last_diagnostics
print("Last Barra diagnostics:", last_diag_barra.as_dict())

last_diag_edge = edge_allocator.last_diagnostics
print("Last Edge diagnostics:", last_diag_edge.as_dict())
