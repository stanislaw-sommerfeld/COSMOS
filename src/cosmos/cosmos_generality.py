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

    print("=== Generality: B = D2 - G, one solver, any linear dynamics (change 2 numbers) ===")
    cwc, ssc = cw_coeffs(nm), ss_coeffs(nm, inc)
    for name, (kxx, kc) in [("free space", (0.0, 0.0)),
                            ("Clohessy-Wiltshire", cwc),
                            ("J2 / Schweighart-Sedwick", ssc)]:
        _, dv = plan(kxx, kc, start, goal, N, dt)
        print(f"{name:26}  kxx={kxx:8.4f}  kc={kc:8.4f}   (energy-optimal plan, dv={dv:.3f})")
    dcoeff = max(abs(cwc[0] - ssc[0]), abs(cwc[1] - ssc[1]))
    print(f"\nCW vs J2 coefficient difference at LEO: {dcoeff:.2e}  -> the two plans coincide")
    print("(CW is adequate for short LEO transfers). J2 matters over long horizons.")

    # ========================================================================
    # Two panels: (L) the construction accepts ANY linear dynamics -- shown with
    # three GENUINELY distinct G, so the point is visible; (R) the honest LEO
    # zoom, CW vs J2, which coincide -- annotated so it reads as the claim, not a
    # glitch.
    # ========================================================================
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.8))

    # --- Left: distinct dynamics (vary the radial stiffness / Coriolis freely) ---
    # These are illustrative linear dynamics (not all physical) whose only purpose
    # is to show that plugging a different G gives a different optimal plan with
    # the SAME solver. CW is one of them.
    demo = [("free space            (kxx, kc)=(0, 0)",        (0.0, 0.0),          "#8b97a8", "-"),
            ("Clohessy-Wiltshire    (3n², 2n)",               cwc,                 "#2563EB", "-"),
            ("stiffer coupling      (6n², 3n)",               (6.0 * nm**2, 3*nm), "#1d9e75", "-"),
            ("retrograde coupling   (3n², -2n)",              (3.0 * nm**2, -2*nm),"#DC2626", "-")]
    for name, (kxx, kc), color, ls in demo:
        traj, dv = plan(kxx, kc, start, goal, N, dt)
        Q = np.vstack([start, traj, goal])
        axL.plot(Q[:, 1], Q[:, 0], ls, color=color, lw=2.6, label=f"{name}  ·  Δv={dv:.2f}")
    axL.scatter(start[1], start[0], c="k", s=55, zorder=5)
    axL.scatter(goal[1], goal[0], c="k", marker="*", s=200, zorder=5)
    axL.annotate("start", (start[1], start[0]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    axL.annotate("goal", (goal[1], goal[0]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    axL.set_xlabel("along-track  y"); axL.set_ylabel("radial  x")
    axL.set_aspect("equal"); axL.grid(alpha=0.3); axL.legend(fontsize=8.3, loc="lower left")
    axL.set_title("Plug in ANY linear dynamics G — the optimal plan changes,\n"
                  "the solver does not (only two numbers change)", fontsize=10.5)

    # --- Right: the honest LEO case, CW vs J2 (coincide) ---
    tcw, dvcw = plan(*cwc, start, goal, N, dt)
    tj2, dvj2 = plan(*ssc, start, goal, N, dt)
    Qcw = np.vstack([start, tcw, goal]); Qj2 = np.vstack([start, tj2, goal])
    axR.plot(Qcw[:, 1], Qcw[:, 0], "-", color="#2563EB", lw=4.0, alpha=0.9,
             label=f"Clohessy-Wiltshire  (Δv={dvcw:.3f})")
    axR.plot(Qj2[:, 1], Qj2[:, 0], "--", color="#DC2626", lw=2.0,
             label=f"J2 / Schweighart-Sedwick  (Δv={dvj2:.3f})")
    axR.scatter(start[1], start[0], c="k", s=55, zorder=5)
    axR.scatter(goal[1], goal[0], c="k", marker="*", s=200, zorder=5)
    axR.set_xlabel("along-track  y"); axR.set_ylabel("radial  x")
    axR.set_aspect("equal"); axR.grid(alpha=0.3); axR.legend(fontsize=9, loc="lower left")
    axR.set_title("At LEO, CW and J2 nearly coincide — and that is the point:\n"
                  "CW is the right, cheap model for short LEO reconfiguration", fontsize=10.5)
    axR.text(0.97, 0.96, f"the two curves overlap\n(coefficients differ by {dcoeff:.0e})",
             transform=axR.transAxes, ha="right", va="top", fontsize=8.5, color="#555",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f7fb", ec="#dfe3ea"))

    fig.suptitle("COSMOS — one construction, any linear dynamics  (B = D₂ − G)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "Only the two coefficients (kxx, kc) of G change between dynamics; the metric BᵀB, "
             "its preconditioner and the solver are identical.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("cosmos_generality.png", dpi=150)
    print("\nFigure -> cosmos_generality.png")


if __name__ == "__main__":
    main()
