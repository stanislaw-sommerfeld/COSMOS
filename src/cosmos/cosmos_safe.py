"""
COSMOS - Module B: metric-aware safe optimization (Riemannian projection)
=========================================================================

Module A exposed the "faille": enforcing a keep-out with a metric-BLIND Euclidean
projection injects ~15x the optimal energy. Module B fixes it by projecting in the
fuel metric M = B^T B (a Riemannian projection), so strict safety no longer erodes
fuel.

For one linearized keep-out constraint at waypoint i (outward normal n_i):
    a^T xi >= b,   a = (n_i on the i-th x,y slots),  b = r + margin + n_i^T c
the minimum-FUEL correction (min  d xi^T M d xi  s.t. feasible) is closed form:
    d xi = M^-1 a (b - a^T xi) / (a^T M^-1 a)        [Riemannian]
vs the metric-blind one that moves a single waypoint radially:
    x_i  <- c + n_i (r + margin)                     [Euclidean]

Refactor note: the LQ optimum and the two projections now come from
``cosmos._core``; the experiment is wrapped in ``run()`` / ``main()`` with a
``__main__`` guard so the module imports cleanly and is unit-testable.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import (cw_operators, cw_offset, to_flat, to_traj,
                      obstacle_gradient, cw_control)
from ._core import (cw_state_matrices, lq_optimal_trajectory,
                    project_euclid, project_metric)


def lq_opt(s0, sf, A, B, T):
    """Closed-form min-energy transfer (J*, position trajectory). Wrapper over
       cosmos._core.lq_optimal_trajectory."""
    return lq_optimal_trajectory(s0, sf, A, B, T, steps=400)


def energy(traj, start, goal, dt, nm):
    U = cw_control(traj, start, goal, dt, nm)
    return np.sum(np.linalg.norm(U, axis=1) ** 2) * dt


def clearance(traj, start, goal, obs):
    Q = np.vstack([start, traj, goal])
    return np.min(np.linalg.norm(Q - obs["c"], axis=1)) - obs["r"]


def solve(start, goal, nm, n, dt, obs, w_bc, eps, step, iters, mode):
    """Fuel-aware CHOMP with a hard keep-out projection: ``mode='euclid'``
       (metric-blind) or ``mode='riemann'`` (Riemannian, B^T B)."""
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc)
    c0 = cw_offset(blocks, start, goal)
    traj = np.linspace(start, goal, n + 2)[1:-1].copy()
    for _ in range(iters):
        xi = to_flat(traj)
        g = B.T @ (B @ xi + c0)
        g_o, _ = obstacle_gradient(traj, start, goal, dt, obs["c"], obs["r"], eps)
        traj = to_traj(xi - step * (Minv @ (g + to_flat(g_o))), n)
        if mode == "euclid":
            traj = project_euclid(traj, [obs])
        else:
            traj = project_metric(traj, [obs], Minv)
    return traj


def run(nm=1.0, N=60, T=np.pi, w_bc=20000.0, verbose=True):
    """Run Module B and return a results dict (no print unless ``verbose``)."""
    A, B = cw_state_matrices(nm, dim=2)
    dt = T / (N + 1)
    start = np.array([3.0, 5.0]); goal = np.array([0.0, 0.0])
    obs = {"c": np.array([2.5, 2.9]), "r": 1.1}
    Jstar, lq_traj = lq_opt(np.array([3., 5., 0, 0]), np.zeros(4), A, B, T)

    tj_eu = solve(start, goal, nm, N, dt, obs, w_bc, 0.8, 0.02, 2000, "euclid")
    tj_ri = solve(start, goal, nm, N, dt, obs, w_bc, 0.8, 0.02, 2000, "riemann")
    E_eu = energy(tj_eu, start, goal, dt, nm)
    E_ri = energy(tj_ri, start, goal, dt, nm)

    if verbose:
        print("=== Module B: metric-aware vs metric-blind keep-out projection ===")
        print(f"unconstrained LQ optimum J*      : {Jstar:.3f}")
        print(f"Euclidean (metric-blind) proj.   : clearance {clearance(tj_eu,start,goal,obs):+.3f}"
              f"  energy {E_eu:.1f}  ({E_eu/Jstar:.1f}x optimal)")
        print(f"Riemannian (B^T B-metric) proj.  : clearance {clearance(tj_ri,start,goal,obs):+.3f}"
              f"  energy {E_ri:.1f}  ({E_ri/Jstar:.2f}x optimal)")
        print(f"-> Module B cuts the safety premium from {E_eu/Jstar:.0f}x to "
              f"{E_ri/Jstar:.2f}x  ({100*(E_eu-E_ri)/E_eu:.0f}% energy saved while staying feasible)")

    return dict(start=start, goal=goal, obs=obs, Jstar=Jstar, lq_traj=lq_traj,
                tj_eu=tj_eu, tj_ri=tj_ri, E_eu=E_eu, E_ri=E_ri)


def main(out="cosmos_safe.png"):
    r = run(verbose=True)
    start, goal, obs = r["start"], r["goal"], r["obs"]
    Jstar, lq_traj, E_eu, E_ri = r["Jstar"], r["lq_traj"], r["E_eu"], r["E_ri"]
    tj_eu, tj_ri = r["tj_eu"], r["tj_ri"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    th = np.linspace(0, 2 * np.pi, 120)
    ax1.fill(obs["c"][1] + obs["r"] * np.cos(th), obs["c"][0] + obs["r"] * np.sin(th),
             color="#d96459", alpha=0.15)
    ax1.plot(lq_traj[:, 1], lq_traj[:, 0], color="k", lw=2.5, alpha=0.4,
             label=f"LQ optimum (violates)")
    Qe = np.vstack([start, tj_eu, goal]); Qr = np.vstack([start, tj_ri, goal])
    ax1.plot(Qe[:, 1], Qe[:, 0], "-", color="#d96459", lw=2.2,
             label=f"Euclidean proj. ({E_eu/Jstar:.0f}x)")
    ax1.plot(Qr[:, 1], Qr[:, 0], "-", color="#1d9e75", lw=2.6,
             label=f"Riemannian proj. ({E_ri/Jstar:.2f}x)")
    ax1.scatter(start[1], start[0], c="k", zorder=5)
    ax1.scatter(goal[1], goal[0], c="r", marker="*", s=140, zorder=5)
    ax1.set_xlabel("along-track  y"); ax1.set_ylabel("radial  x")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    ax1.set_title("Same keep-out, two projections")

    bars = ["LQ optimum\n(unconstrained)", "Riemannian\n(Module B)", "Euclidean\n(naive)"]
    vals = [Jstar, E_ri, E_eu]; cols = ["k", "#1d9e75", "#d96459"]
    ax2.bar(bars, vals, color=cols); ax2.set_yscale("log")
    ax2.set_ylabel("control energy (log)")
    for i, v in enumerate(vals):
        ax2.text(i, v * 1.05, f"{v/Jstar:.2f}x", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Safety premium: 15x -> near-optimal")
    fig.suptitle("COSMOS - Module B: metric-aware safe optimization "
                 "(Riemannian projection in B^T B)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
