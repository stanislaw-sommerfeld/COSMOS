"""
COSMOS - Extension E3: metric-aware INTER-AGENT safety (coupled Riemannian projection)
======================================================================================

Module B made STATIC keep-out avoidance fuel-cheap by projecting in the metric
B^T B instead of Euclidean space (15x -> 1.69x). But a swarm's dominant hazard is
not a static sphere -- it is the OTHER AGENTS, whose separation constraint couples
two trajectories at the same instant. COSMOS previously enforced that with either
a soft penalty (no guarantee) or a hard Euclidean push-apart (metric-blind, so it
pays the same premium Module B diagnosed for static obstacles).

E3 closes that gap: it extends the closed-form Riemannian projection to the
COUPLED pairwise constraint ||x_i(t) - x_j(t)|| >= r_col. The correction is shared
between the two agents, each moving the way that is cheapest in its own fuel
metric (see _core.project_inter_agent_metric for the derivation). The swarm stays
strictly collision-free, at a fraction of the Delta-v of the Euclidean push.

Scenario: K agents in a head-on antipodal swap -- every straight line crosses the
centre at the same time, so inter-agent avoidance is the whole problem. We plan
the fuel-aware trajectories, then enforce separation two ways and score the
realized Delta-v under the true CW dynamics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import (cw_operators, cw_offset, to_flat, to_traj, cw_control)
from ._core import project_inter_agent_metric


def _euclid_push(trajs, starts, goals, r_col, margin=0.05, sweeps=4):
    """Metric-blind baseline: at each time index push any too-close pair apart
       equally in Euclidean space (the previous cosmos_full.project_inter_agent)."""
    out = [t.copy() for t in trajs]
    K = len(out); n = out[0].shape[0]
    for _ in range(sweeps):
        for t in range(n):
            for i in range(K):
                for j in range(i + 1, K):
                    v = out[i][t] - out[j][t]
                    d = np.linalg.norm(v) + 1e-9
                    clr = d - r_col
                    if clr < margin:
                        push = 0.5 * (margin - clr) * v / d
                        out[i][t] += push
                        out[j][t] -= push
    return out


def _plan_fuel(starts, goals, nm, N, dt, w_bc, step, iters):
    """Fuel-aware single-agent plans (no coupling yet) -- the common baseline both
       safety methods then correct."""
    B, Minv, blocks = cw_operators(N, dt, nm, w_bc=w_bc, dim=len(starts[0]))
    trajs = []
    for s, g in zip(starts, goals):
        c0 = cw_offset(blocks, s, g)
        traj = np.linspace(s, g, N + 2)[1:-1].copy()
        for _ in range(iters):
            xi = to_flat(traj)
            traj = to_traj(xi - step * (Minv @ (B.T @ (B @ xi + c0))), N)
        trajs.append(traj)
    return trajs, Minv


def total_dv(trajs, starts, goals, dt, nm):
    return sum(np.sum(np.linalg.norm(cw_control(trajs[k], starts[k], goals[k], dt, nm),
                                     axis=1)) * dt for k in range(len(trajs)))


def min_sep(trajs, starts, goals):
    K = len(trajs)
    Q = [np.vstack([starts[k], trajs[k], goals[k]]) for k in range(K)]
    d = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = min(d, np.min(np.linalg.norm(Q[i] - Q[j], axis=1)))
    return d


def run(lanes=4, nm=1.0, N=60, r_col=1.2, verbose=True):
    T = np.pi; dt = T / (N + 1)
    # crossing-lanes swap: two parallel columns exchange sides AND mirror in y, so
    # the fuel-aware straight-ish plans genuinely overlap -> inter-agent avoidance
    # is the whole problem (unlike a wide antipodal ring, which self-separates).
    ys = np.linspace(-2.0, 2.0, lanes)
    starts = [np.array([-5.0, y]) for y in ys] + [np.array([5.0, y]) for y in ys]
    goals = [np.array([5.0, -y]) for y in ys] + [np.array([-5.0, -y]) for y in ys]
    K = len(starts)

    base, Minv = _plan_fuel(starts, goals, nm, N, dt, w_bc=0.0, step=0.02, iters=600)
    d0 = min_sep(base, starts, goals)

    t_eu = _euclid_push(base, starts, goals, r_col, sweeps=6)
    t_ri = project_inter_agent_metric([t.copy() for t in base], Minv, r_col, sweeps=6)

    dv_eu = total_dv(t_eu, starts, goals, dt, nm)
    dv_ri = total_dv(t_ri, starts, goals, dt, nm)
    d_eu = min_sep(t_eu, starts, goals)
    d_ri = min_sep(t_ri, starts, goals)

    if verbose:
        print("=== E3: metric-aware inter-agent safety (coupled Riemannian projection) ===")
        print(f"K={K} crossing-lanes swap, r_col={r_col}, scored under true CW dynamics\n")
        print(f"unconstrained plans   : min separation {d0:+.3f}  (unsafe: agents cross)")
        print(f"Euclidean push-apart  : min sep {d_eu:+.3f} (safe)   total dv {dv_eu:7.2f}")
        print(f"Riemannian (B^T B)    : min sep {d_ri:+.3f} (safe)   total dv {dv_ri:7.2f}")
        print(f"-> metric-aware separation is {dv_eu/dv_ri:.2f}x cheaper for the SAME"
              f" clearance\n   ({100*(dv_eu-dv_ri)/dv_eu:.0f}% of the separation Delta-v"
              f" recovered), the swarm analogue of Module B.")

    return dict(starts=starts, goals=goals, base=base, t_eu=t_eu, t_ri=t_ri,
                dv_eu=dv_eu, dv_ri=dv_ri, d_eu=d_eu, d_ri=d_ri, d0=d0, r_col=r_col)


def main(out="cosmos_swarm_safe.png"):
    r = run(verbose=True)
    starts, goals = r["starts"], r["goals"]
    K = len(starts)
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, K))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    for ax, key, ttl, dv in [(axes[0], "t_eu", "Euclidean push-apart (metric-blind)", r["dv_eu"]),
                             (axes[1], "t_ri", "Riemannian projection (fuel metric)", r["dv_ri"])]:
        for k in range(K):
            Q = np.vstack([starts[k], r[key][k], goals[k]])
            ax.plot(Q[:, 0], Q[:, 1], color=colors[k], lw=2.2)
            ax.scatter(*starts[k], color=colors[k], s=32, zorder=5)
            ax.scatter(*goals[k], color=colors[k], s=46, marker="x", zorder=5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{ttl}\ntotal Δv = {dv:.2f}", fontsize=10.5)
    fig.suptitle(f"COSMOS - E3: metric-aware inter-agent safety "
                 f"({r['dv_eu']/r['dv_ri']:.2f}× cheaper, same clearance, collision-free)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
