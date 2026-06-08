"""
COSMOS - Generality of the construction: B = D2 - G, plug in any linear dynamics
================================================================================

The whole point of COSMOS in one figure: the planner is NOT hard-wired to Clohessy-
Wiltshire. Its metric is B = D2 - G; CW is one choice of the relative dynamics G.
Here the SAME covariant construction and the SAME closed-form optimum plan a transfer
under three dynamics, changing only two coefficients (radial stiffness kxx, Coriolis
rate kc):

    free space         (kxx, kc) = (0, 0)            -> straight line
    Clohessy-Wiltshire (kxx, kc) = (3 n^2, 2 n)      -> orbital arc
    J2 / Schweighart-Sedwick      = ((5c^2-2)n^2, 2nc)

HONEST REGIME NOTE: at LEO over a short transfer (~half to one orbit), c is ~1.00005,
so the J2 plan and the CW plan are visually identical -- which is exactly why CW is an
excellent model for a short *reconfiguration*. J2's payoff is over LONG horizons
(formation *maintenance*), a different problem (see cw_vs_j2_demo.py). This figure
shows the *architecture* generalizes, not that J2 changes a short maneuver.

So COSMOS can claim "energy-aware planning for any linear relative dynamics" while its
title and headline results stay squarely about the CW instance. This file is the proof.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import (linear_relative_operators, linear_relative_offset,
                       cw_coeffs, ss_coeffs, to_traj)


def plan(kxx, kc, start, goal, n, dt, w_bc=20000.0):
    """Closed-form energy-optimal transfer for the given linear dynamics:
       xi* = -(B^T B)^-1 B^T c0  (the no-obstacle optimum the descent converges to)."""
    B, Minv, blocks = linear_relative_operators(n, dt, kxx, kc, w_bc=w_bc)
    c0 = linear_relative_offset(blocks, start, goal)
    xi = -Minv @ (B.T @ c0)
    u = B @ xi + c0
    dv = sum(np.hypot(u[i], u[n + i]) for i in range(n)) * dt   # L1 control under this G
    return to_traj(xi, n), dv


def main():
    nm = 1.0
    inc = np.deg2rad(51.6)
    T = 1.5 * np.pi; N = 60; dt = T / (N + 1)
    start = np.array([3.0, 5.0]); goal = np.array([0.0, 0.0])

    cases = [("free space",            (0.0, 0.0),            "#8b97a8", "-"),
             ("Clohessy–Wiltshire",    cw_coeffs(nm),         "#2563EB", "-"),
             ("J2 / Schweighart–Sedwick", ss_coeffs(nm, inc), "#DC2626", "--")]

    print("=== Generality: B = D2 - G, one solver, three dynamics (change 2 numbers) ===")
    plans = []
    for name, (kxx, kc), color, ls in cases:
        traj, dv = plan(kxx, kc, start, goal, N, dt)
        plans.append((name, traj, color, ls, dv))
        print(f"{name:26}  kxx={kxx:7.4f}  kc={kc:7.4f}   (energy-optimal plan, dv={dv:.3f})")

    cwc, ssc = cw_coeffs(nm), ss_coeffs(nm, inc)
    dcoeff = max(abs(cwc[0]-ssc[0]), abs(cwc[1]-ssc[1]))
    print(f"\nCW vs J2 coefficient difference: {dcoeff:.2e}  -> the two plans coincide at")
    print("LEO over a short transfer (CW is adequate here). J2 matters over long horizons")
    print("(formation maintenance) -- future work. The construction itself is unchanged.")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    for name, traj, color, ls, dv in plans:
        Q = np.vstack([start, traj, goal])
        ax.plot(Q[:, 1], Q[:, 0], ls, color=color, lw=2.6 if ls == "-" else 2.0,
                label=f"{name}", alpha=0.95)
    ax.scatter(start[1], start[0], c="k", s=55, zorder=5)
    ax.scatter(goal[1], goal[0], c="k", marker="*", s=200, zorder=5)
    ax.annotate("start", (start[1], start[0]), textcoords="offset points",
                xytext=(8, 6), fontsize=9)
    ax.annotate("goal", (goal[1], goal[0]), textcoords="offset points",
                xytext=(8, 6), fontsize=9)
    ax.set_xlabel("along-track  y"); ax.set_ylabel("radial  x")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")
    ax.set_title("COSMOS — one construction, any linear dynamics (B = D₂ − G)\n"
                 "free vs CW differ visibly; CW vs J2 coincide at LEO (CW is adequate here)",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01,
             "Only two coefficients (kxx, kc) change between dynamics — the metric BᵀB, "
             "preconditioner and solver are identical. That is the generality claim, in code.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("cosmos_generality.png", dpi=130)
    print("\nFigure -> cosmos_generality.png")


if __name__ == "__main__":
    main()
