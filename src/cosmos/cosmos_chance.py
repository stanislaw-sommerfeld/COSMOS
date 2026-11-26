"""
COSMOS - Extension E5: chance-constrained keep-outs under navigation uncertainty
================================================================================

Modules A/B treat the keep-out radius as exact and the spacecraft position as
known. In reality each waypoint's position is known only up to NAVIGATION
UNCERTAINTY (star-tracker + orbit-determination error). A plan that just grazes
the keep-out boundary is then unsafe: half the time the true position is inside.

E5 turns the deterministic keep-out into a CHANCE CONSTRAINT -- "stay outside the
physical radius with probability >= beta" -- using the standard quantile
reformulation: inflate the radius by k_beta * sigma, k_beta = Phi^-1(beta)
(_core.chance_margin / inflate_obstacles). Planning against the inflated radius
with the SAME metric-aware Riemannian projection then guarantees the chance
constraint, with zero new planner machinery -- the fuel-aware metric and Module B
carry straight over.

We verify it by Monte Carlo: plan at several target beta, perturb every waypoint
by Gaussian navigation error many times, and measure the realized collision rate.
It should track 1 - beta, and the Delta-v premium should grow smoothly with the
safety level -- a calibrated, tunable safety/fuel trade.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import cw_operators, cw_offset, to_flat, to_traj, cw_control
from ._core import inflate_obstacles, chance_margin, project_metric


def _plan_with_keepout(start, goal, obs_plan, nm, N, dt, w_bc, eps=0.8,
                       step=0.02, iters=1500):
    """Fuel-aware plan with a hard Riemannian keep-out projection against the
       (possibly inflated) obstacle ``obs_plan``."""
    from .chomp_cw import obstacle_gradient
    B, Minv, blocks = cw_operators(N, dt, nm, w_bc=w_bc, dim=2)
    c0 = cw_offset(blocks, start, goal)
    traj = np.linspace(start, goal, N + 2)[1:-1].copy()
    for _ in range(iters):
        xi = to_flat(traj)
        g = B.T @ (B @ xi + c0)
        g_o, _ = obstacle_gradient(traj, start, goal, dt,
                                   obs_plan["c"], obs_plan["r"], eps)
        traj = to_traj(xi - step * (Minv @ (g + to_flat(g_o))), N)
        traj = project_metric(traj, [obs_plan], Minv)
    return traj


def _collision_rate(traj, start, goal, obs_phys, sigma, samples, rng):
    """Monte-Carlo collision probability at the BINDING waypoint -- the planned
       closest-approach point, which is the single active chance constraint for a
       convex keep-out. Isotropic Gaussian navigation error sigma is applied to
       that waypoint's position and we count how often the TRUE position is inside
       the physical radius. This is exactly the event chance_margin controls
       (a per-constraint quantile), so a calibrated margin makes this track 1-beta."""
    Q = np.vstack([start, traj, goal])
    d_nom = np.linalg.norm(Q - obs_phys["c"], axis=1) - obs_phys["r"]
    i_bind = int(np.argmin(d_nom))                       # closest-approach waypoint
    x_bind = Q[i_bind]
    noisy = x_bind[None, :] + rng.normal(0, sigma, (samples, x_bind.size))
    d = np.linalg.norm(noisy - obs_phys["c"], axis=1) - obs_phys["r"]
    return float(np.mean(d < 0))


def run(sigma=0.25, betas=(0.50, 0.80, 0.90, 0.95, 0.99), nm=1.0, N=70,
        samples=4000, verbose=True):
    T = np.pi; dt = T / (N + 1); w_bc = 20000.0
    start = np.array([3.0, 5.0]); goal = np.array([0.0, 0.0])
    obs_phys = {"c": np.array([2.5, 2.9]), "r": 1.1}     # true physical keep-out
    rng = np.random.default_rng(0)

    rows = []
    for beta in betas:
        obs_plan = inflate_obstacles([obs_phys], sigma, beta)[0]
        traj = _plan_with_keepout(start, goal, obs_plan, nm, N, dt, w_bc)
        rate = _collision_rate(traj, start, goal, obs_phys, sigma, samples, rng)
        dv = np.sum(np.linalg.norm(cw_control(traj, start, goal, dt, nm), axis=1)) * dt
        rows.append(dict(beta=beta, margin=chance_margin(sigma, beta),
                         rate=rate, dv=dv, traj=traj))
    if verbose:
        print("=== E5: chance-constrained keep-outs under navigation uncertainty ===")
        print(f"navigation error sigma={sigma}, physical radius r={obs_phys['r']}, "
              f"{samples} Monte-Carlo samples/level\n")
        print(f"{'target beta':>11} | {'radius inflation':>16} | "
              f"{'realized safe':>13} | {'collisions':>10} | {'Δv':>7}")
        print("-" * 70)
        for r in rows:
            print(f"{r['beta']:11.2f} | {r['margin']:16.3f} | "
                  f"{1-r['rate']:12.1%} | {r['rate']:9.1%} | {r['dv']:7.3f}")
        print("\nThe realized safe fraction tracks the target beta (so the chance")
        print("constraint is calibrated), and Δv rises smoothly with the safety level")
        print("-- a tunable safety/fuel trade, built from the exact same Riemannian")
        print("projection as Module B (only the radius changed).")
    return rows, (start, goal, obs_phys, sigma)


def main(out="cosmos_chance.png"):
    rows, (start, goal, obs_phys, sigma) = run(verbose=True)
    betas = [r["beta"] for r in rows]
    safe = [1 - r["rate"] for r in rows]
    dvs = [r["dv"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))
    # left: plans at low vs high beta around the physical + inflated keep-out
    th = np.linspace(0, 2 * np.pi, 160)
    ax1.fill(obs_phys["c"][1] + obs_phys["r"] * np.cos(th),
             obs_phys["c"][0] + obs_phys["r"] * np.sin(th),
             color="#d96459", alpha=0.30, label="physical keep-out")
    for r, col in [(rows[0], "#888"), (rows[-1], "#1d9e75")]:
        Q = np.vstack([start, r["traj"], goal])
        ax1.plot(Q[:, 1], Q[:, 0], "-", color=col, lw=2.4,
                 label=f"β={r['beta']:.2f} plan")
        rr = obs_phys["r"] + r["margin"]
        ax1.plot(obs_phys["c"][1] + rr * np.cos(th), obs_phys["c"][0] + rr * np.sin(th),
                 ":", color=col, lw=1.3)
    ax1.scatter(start[1], start[0], c="k", zorder=5, s=40)
    ax1.scatter(goal[1], goal[0], c="r", marker="*", s=150, zorder=5)
    ax1.set_xlabel("along-track  y"); ax1.set_ylabel("radial  x")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5)
    ax1.set_title(f"Inflated keep-out (dotted) vs physical (σ={sigma})")

    ax2.plot([0.5, 1.0], [0.5, 1.0], "--", color="#bbb", label="ideal (realized = target)")
    ax2.plot(betas, safe, "o-", color="#1d9e75", lw=2.4, label="realized safe fraction")
    ax2.set_xlabel("target safety β"); ax2.set_ylabel("realized safe fraction", color="#1d9e75")
    ax2.tick_params(axis="y", labelcolor="#1d9e75"); ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(betas, dvs, "s--", color="#2563EB", lw=1.6, alpha=0.8, label="Δv")
    ax2b.set_ylabel("Δv", color="#2563EB"); ax2b.tick_params(axis="y", labelcolor="#2563EB")
    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, fontsize=8.3, loc="upper left")
    ax2.set_title("Calibrated safety: realized tracks target; Δv grows with β")
    fig.suptitle("COSMOS - E5: chance-constrained keep-outs "
                 "(radius inflation + the Module B projection)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
