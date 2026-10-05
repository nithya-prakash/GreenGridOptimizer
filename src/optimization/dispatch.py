"""Battery dispatch that firms the renewable forecast toward a flat block.

Given hourly forecast generation g_t (MW), schedule a battery (charge c_t,
discharge d_t, state of charge s_t, ending at least as full as it started) so the delivered power g_t - c_t + d_t stays
as close as possible to the forecast mean b, minimising the total absolute
deviation sum_t |g_t - c_t + d_t - b|. Solved as a linear program.

Scope: this is physical firming of a forecast, not market arbitrage (no prices
are modelled). It optimises against the *forecast*, so the real gain depends on
forecast accuracy; the forecast's own prediction intervals are the right way to
judge the risk.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    power_mw: float          # max charge and discharge rate
    capacity_mwh: float      # usable energy capacity
    round_trip_efficiency: float = 0.90
    initial_soc_fraction: float = 0.5

    def __post_init__(self):
        if self.power_mw <= 0 or self.capacity_mwh <= 0:
            raise ValueError("power_mw and capacity_mwh must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if not 0 <= self.initial_soc_fraction <= 1:
            raise ValueError("initial_soc_fraction must be in [0, 1]")


def optimize_dispatch(generation: pd.Series, battery: Battery, block_mw: float | None = None) -> dict:
    """`generation`: hourly forecast MW. `block_mw`: delivery target, default the mean.
    Returns the hourly schedule and the deviation before and after."""
    g = generation.to_numpy(dtype=float)
    T = len(g)
    if T == 0:
        raise ValueError("generation is empty")
    block = float(g.mean()) if block_mw is None else float(block_mw)
    eta = battery.round_trip_efficiency ** 0.5  # split losses evenly between charge and discharge
    s0 = battery.initial_soc_fraction * battery.capacity_mwh

    # Variables: c[0:T], d[0:T], u[0:T], v[0:T]  (u, v = positive / negative deviation from the block)
    n = 4 * T
    c_idx, d_idx, u_idx, v_idx = (np.arange(T) + k * T for k in range(4))
    cost = np.zeros(n)
    cost[u_idx] = 1.0
    cost[v_idx] = 1.0

    # Delivered_t - block = u_t - v_t  ->  -c_t + d_t - u_t + v_t = block - g_t
    A_eq = np.zeros((T, n))
    A_eq[np.arange(T), c_idx] = -1.0
    A_eq[np.arange(T), d_idx] = 1.0
    A_eq[np.arange(T), u_idx] = -1.0
    A_eq[np.arange(T), v_idx] = 1.0
    b_eq = block - g

    # SOC_t = s0 + sum_{k<=t} (eta*c_k - d_k/eta), kept within [0, capacity].
    tri = np.tril(np.ones((T, T)))
    soc_coeff = np.zeros((T, n))
    soc_coeff[:, c_idx] = eta * tri
    soc_coeff[:, d_idx] = -tri / eta
    # Energy-neutral over the horizon: the battery must end at least as full as it started,
    # so the initial charge is not counted as free energy.
    A_ub = np.vstack([soc_coeff, -soc_coeff, -soc_coeff[-1:]])
    b_ub = np.concatenate([np.full(T, battery.capacity_mwh - s0), np.full(T, s0), [0.0]])

    bounds = [(0, battery.power_mw)] * (2 * T) + [(0, None)] * (2 * T)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"dispatch LP failed: {res.message}")

    charge, discharge = res.x[c_idx], res.x[d_idx]
    soc = s0 + np.cumsum(eta * charge - discharge / eta)
    schedule = pd.DataFrame({"generation_mw": g, "charge_mw": charge, "discharge_mw": discharge,
                             "soc_mwh": soc, "delivered_mw": g - charge + discharge}, index=generation.index)
    before = float(np.abs(g - block).sum())
    after = float(np.abs(schedule["delivered_mw"] - block).sum())
    return {
        "schedule": schedule,
        "block_mw": block,
        "deviation_mwh_before": before,
        "deviation_mwh_after": after,
        "deviation_reduction_pct": 0.0 if before == 0 else 100.0 * (before - after) / before,
        "energy_lost_mwh": float(schedule["charge_mw"].sum() - schedule["discharge_mw"].sum()
                                 - (soc[-1] - s0)),
    }
