"""
COSMOS - Module A: optimal-control benchmark (LQ / controllability Gramian)
===========================================================================

The CW dynamics are LINEAR and our fuel cost is QUADRATIC, so the minimum-energy
rendezvous is a Linear-Quadratic problem with a CLOSED-FORM optimum:

    state  s = [x, y, vx, vy],   s_dot = A s + B u
    min-energy transfer s0 -> sf in time T:
        Wr(T) = int_0^T Phi(sigma) B B^T Phi(sigma)^T dsigma   (reachability Gramian)
        J*    = (sf - Phi(T) s0)^T Wr(T)^-1 (sf - Phi(T) s0)   (optimal energy)
        u*(t) = B^T Phi(T-t)^T Wr(T)^-1 (sf - Phi(T) s0)

We use J* as GROUND TRUTH to show:
  (1) fuel-aware CHOMP recovers ~the optimal transfer (validation),
  (2) free-space CHOMP is far above optimal (the metric matters),
  (3) CHOMP also handles a keep-out that the LQ optimum CANNOT (its value-add).

Energies are dimensionless ratios (scale-invariant), so we work in normalized
units (n=1, T=pi), matching the rest of the project.

Refactor note: the Gramian / LQ optimum / CW state-system now come from
``cosmos._core`` (a single shared implementation). The experiment is wrapped in
``run()`` / ``main()`` with a ``__main__`` guard, so the module can be imported
and unit-tested without executing the whole demo as a side effect.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import (chomp_cw, cw_control, cw_operators, cw_offset,
                      to_flat, to_traj, obstacle_gradient)
from ._core import (cw_state_matrices, controllability_gramian as reachability_gramian,
                    lq_optimal_trajectory)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def lq_optimal(s0, sf, A, B, T):
    """Closed-form min-energy transfer. Returns J*, and the optimal trajectory.
       Thin wrapper over cosmos._core.lq_optimal_trajectory (kept for API
       compatibility and the 2D position-trajectory reconstruction)."""
    return lq_optimal_trajectory(s0, sf, A, B, T, steps=400)


def chomp_energy(traj, start, goal, dt, nm):
    U = cw_control(traj, start, goal, dt, nm)
    return np.sum(np.linalg.norm(U, axis=1) ** 2) * dt


def chomp_cw_avoid(start, goal, nm, n, dt, obs, w_bc, eps, step, iters, margin=0.05):
    """Fuel-aware CHOMP with a HARD keep-out projection (Euclidean = metric-blind;
       Module B will replace this projection with a B^T B (Riemannian) one)."""
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc)
    c0 = cw_offset(blocks, start, goal)
    traj = np.linspace(start, goal, n + 2)[1:-1].copy()
    for _ in range(iters):
        xi = to_flat(traj)
        g = B.T @ (B @ xi + c0)
        g_o, _ = obstacle_gradient(traj, start, goal, dt, obs["c"], obs["r"], eps)
        g = g + to_flat(g_o)
        traj = to_traj(xi - step * (Minv @ g), n)
        for i in range(n):                       # hard keep-out projection
            v = traj[i] - obs["c"]; d = np.linalg.norm(v)
            if d - obs["r"] < margin:
                traj[i] = obs["c"] + v / (d + 1e-9) * (obs["r"] + margin)
    return traj


def min_clearance(points, center, radius):
    return np.min(np.linalg.norm(points - center, axis=1)) - radius


# ---------------------------------------------------------------------------
# the experiment (importable: returns its results instead of running at import)
# ---------------------------------------------------------------------------
def run(nm=1.0, N=60, T=np.pi, w_bc=20000.0, verbose=True):
    """Run Module A and return a results dict. No file / print side effects
       unless ``verbose``; the figure is drawn by ``main`` from these results."""
    A, B = cw_state_matrices(nm, dim=2)
    dt = T / (N + 1)

    start = np.array([3.0, 5.0]); goal = np.array([0.0, 0.0])
    s0 = np.array([start[0], start[1], 0.0, 0.0])     # at rest
    sf = np.array([goal[0], goal[1], 0.0, 0.0])       # rendezvous at rest

    Jstar, lq_traj = lq_optimal(s0, sf, A, B, T)

    tj_fuel, _, _ = chomp_cw(start, goal, nm=nm, n=N, dt=dt, w_bc=w_bc,
                             step=0.02, iters=2000)
    tj_free, _, _ = chomp_cw(start, goal, nm=0.0, n=N, dt=dt, w_bc=w_bc,
                             step=0.02, iters=2000)
    E_fuel = chomp_energy(tj_fuel, start, goal, dt, nm)
    E_free = chomp_energy(tj_free, start, goal, dt, nm)

    obs = {"c": np.array([2.5, 2.9]), "r": 1.1}       # placed ON the LQ path
    tj_fuel_obs = chomp_cw_avoid(start, goal, nm=nm, n=N, dt=dt, obs=obs,
                                 w_bc=w_bc, eps=0.8, step=0.02, iters=2000)
    E_fuel_obs = chomp_energy(tj_fuel_obs, start, goal, dt, nm)
    lq_clear = min_clearance(lq_traj, obs["c"], obs["r"])
    chomp_clear = min_clearance(np.vstack([start, tj_fuel_obs, goal]),
                                obs["c"], obs["r"])

    if verbose:
        print("=== Module A: optimal-control benchmark (energy, normalized) ===")
        print(f"LQ optimal  J*           : {Jstar:.4f}   (ground truth)")
        print(f"CHOMP fuel-aware energy   : {E_fuel:.4f}   ({E_fuel/Jstar:.2f}x optimal)")
        print(f"CHOMP free-space energy   : {E_free:.4f}   ({E_free/Jstar:.2f}x optimal)")
        print("-" * 52)
        print(f"keep-out at {obs['c']}, r={obs['r']} (placed on the optimal path):")
        print(f"  LQ optimal trajectory clearance : {lq_clear:+.3f}  "
              f"({'VIOLATES keep-out' if lq_clear < 0 else 'ok'})  <- LQ has no obstacle notion")
        print(f"  CHOMP avoids it (naive Euclidean projection):")
        print(f"    clearance {chomp_clear:+.3f}, energy {E_fuel_obs:.1f} "
              f"({E_fuel_obs/Jstar:.1f}x optimal)")
        print(f"  => strict avoidance via the metric-BLIND projection is expensive.")
        print(f"     This {E_fuel_obs/Jstar:.0f}x premium is exactly what Module B (Riemannian")
        print(f"     projection in the B^T B metric) is designed to remove.")

    return dict(start=start, goal=goal, obs=obs, Jstar=Jstar, lq_traj=lq_traj,
                tj_fuel=tj_fuel, tj_free=tj_free, tj_fuel_obs=tj_fuel_obs,
                E_fuel=E_fuel, E_free=E_free, E_fuel_obs=E_fuel_obs,
                lq_clear=lq_clear, chomp_clear=chomp_clear)


def main(out="cosmos_optimal.png"):
    r = run(verbose=True)
    start, goal, obs = r["start"], r["goal"], r["obs"]
    Jstar, lq_traj = r["Jstar"], r["lq_traj"]
    E_fuel, E_free, E_fuel_obs = r["E_fuel"], r["E_free"], r["E_fuel_obs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    Qfuel = np.vstack([start, r["tj_fuel"], goal])
    Qfree = np.vstack([start, r["tj_free"], goal])
    Qobs = np.vstack([start, r["tj_fuel_obs"], goal])
    ax1.plot(lq_traj[:, 1], lq_traj[:, 0], color="k", lw=3, alpha=0.5,
             label=f"LQ optimum (E={Jstar:.2f})")
    ax1.plot(Qfuel[:, 1], Qfuel[:, 0], "--", color="#1d9e75", lw=2,
             label=f"CHOMP fuel-aware ({E_fuel/Jstar:.2f}×)")
    ax1.plot(Qfree[:, 1], Qfree[:, 0], ":", color="#888", lw=2,
             label=f"CHOMP free-space ({E_free/Jstar:.0f}×)")
    th = np.linspace(0, 2 * np.pi, 120)
    ax1.fill(obs["c"][1] + obs["r"] * np.cos(th), obs["c"][0] + obs["r"] * np.sin(th),
             color="#d96459", alpha=0.15)
    ax1.plot(Qobs[:, 1], Qobs[:, 0], "-", color="#d96459", lw=2.4,
             label=f"CHOMP + keep-out ({E_fuel_obs/Jstar:.2f}×)")
    ax1.scatter(start[1], start[0], c="k", zorder=5)
    ax1.scatter(goal[1], goal[0], c="r", marker="*", s=140, zorder=5)
    ax1.set_xlabel("along-track  y"); ax1.set_ylabel("radial  x")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5)
    ax1.set_title("Trajectories vs the analytical LQ optimum")

    bars = ["LQ\noptimum", "CHOMP\nfuel-aware", "CHOMP\nfree-space",
            "CHOMP +keep-out\n(naive proj.)"]
    vals = [Jstar, E_fuel, E_free, E_fuel_obs]
    cols = ["k", "#1d9e75", "#888", "#d96459"]
    ax2.bar(bars, vals, color=cols)
    ax2.set_yscale("log")
    ax2.set_ylim(min(vals) * 0.6, max(vals) * 2.6)
    ax2.set_ylabel(r"control energy  $\int \|u\|^{2}\, dt$  (log scale)")
    ax2.set_title("Energy vs the analytical optimum")
    for i, v in enumerate(vals):
        ax2.text(i, v * 1.07, f"{v/Jstar:.2f}×", ha="center", va="bottom", fontsize=9)
    ax2.annotate("Module B target\n(metric-aware projection)",
                 xy=(3, E_fuel_obs), xytext=(2.0, E_fuel_obs * 0.45),
                 fontsize=8, color="#7a2a1a", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#7a2a1a"))

    fig.suptitle("COSMOS - Module A: fuel-aware CHOMP vs the analytical LQ optimum",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
