import atexit, csv, sys, time
sys.path.insert(0, ".")
from akm_hrp.allocators import retail_alpha_ml_mpc as ram

CAPTURE_PATH = "outputs/kelly_diagnostics_capture.csv"
_rows = []
_fields = ["seq","model_tag","wall_time","observations","crowding_active",
           "crowding_rho_bar","crowding_z","crowding_kappa_multiplier",
           "kappa_used","kelly_fraction","mix_size"]

def _dump():
    if not _rows:
        return
    with open(CAPTURE_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_fields)
        w.writeheader()
        w.writerows(_rows)
    print(f"[capture] wrote {len(_rows)} rows to {CAPTURE_PATH}", flush=True)

atexit.register(_dump)
_orig = ram.RetailAlphaMLMPCAllocator._kelly_factor_allocation

def _wrapped(self):
    mix, fraction = _orig(self)
    diag = dict(self.last_kelly_diagnostics or {})
    tag = "crowding" if self.config.ml_kelly_crowding_kappa_enabled else "baseline"
    _rows.append({
        "seq": len(_rows), "model_tag": tag, "wall_time": time.time(),
        "observations": diag.get("observations", float(len(self._factor_return_buffer))),
        "crowding_active": diag.get("crowding_active", ""),
        "crowding_rho_bar": diag.get("crowding_rho_bar", ""),
        "crowding_z": diag.get("crowding_z", ""),
        "crowding_kappa_multiplier": diag.get("crowding_kappa_multiplier", ""),
        "kappa_used": diag.get("kappa_used", ""),
        "kelly_fraction": fraction,
        "mix_size": int(mix.shape[0]) if hasattr(mix, "shape") else 0,
    })
    if len(_rows) % 20 == 0:
        _dump()
    return mix, fraction

ram.RetailAlphaMLMPCAllocator._kelly_factor_allocation = _wrapped
from akm_hrp.cli import compare_models
compare_models.main()
