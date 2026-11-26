"""
COSMOS - Extension: time-varying metric for ECCENTRIC reference orbits
======================================================================

This is the honest, quantitative pay-off of the "metric = dynamics" claim.

The headline results use Clohessy-Wiltshire (CW), which assumes a CIRCULAR
reference orbit. CW is an excellent model at LEO for short reconfigurations --
cosmos_generality / cw_vs_j2 show that the J2 correction there is ~1e-5, so a
J2-aware metric buys essentially nothing. A reviewer rightly asks: then WHEN does
choosing the metric actually matter?

Answer: when the reference orbit is ECCENTRIC. The linearized relative dynamics
about an eccentric orbit (Tschauner-Hempel / Lawden) are TIME-VARYING: the
coupling block G depends on the reference true anomaly. A CW-"passive" relative
orbit then drifts by MANY TIMES its own amplitude in a single revolution. So the
CW metric is genuinely the wrong metric, and planning in the correct time-varying
metric B = D2 - G(t) produces a large, real fuel saving.

What this module does
---------------------
  1. For a given eccentricity e, plan a rest-to-rest relative transfer two ways:
       - CW metric        (circular-orbit assumption, the baseline)
       - TH metric        (the correct time-varying eccentric dynamics)
     Both use the identical covariant construction; only G changes (now G(t)).
  2. Score BOTH plans under the SAME ground truth: the full NONLINEAR relative
     equations of motion about the eccentric orbit, integrated with a stiff-safe
     RK solver. The cost is the realized Delta-v (L1) needed to track each plan.
  3. Sweep e to show the saving grow from ~0 (circular, CW is right) to large
     (eccentric, the time-varying metric wins) -- the "when does the metric
     matter" curve.

The construction, the preconditioner (B^T B)^-1 and the closed-form optimum are
all unchanged from the rest of COSMOS; only the dynamics block is now G(t). That
is precisely the generality the project claims, demonstrated where it pays.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import to_traj, to_flat
from ._core import (time_varying_operators, time_varying_offset,
                    tschauner_hempel_coeffs)

MU = 1.0        # normalized gravitational parameter
A_REF = 1.0     # normalized reference semi-major axis
NM = np.sqrt(MU / A_REF ** 3)   # reference mean motion (=1 in these units)
PERIOD = 2 * np.pi / NM


# ---------------------------------------------------------------------------
# 1. Reference orbit kinematics (Kepler) and the TRUE nonlinear relative EOM
# ---------------------------------------------------------------------------
def reference_kinematics(t, e):
    """(r, theta_dot, theta_ddot) of the eccentric reference orbit at time t."""
    M = NM * t
    E = M
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))
    r = A_REF * (1.0 - e * np.cos(E))
    h = np.sqrt(MU * A_REF * (1.0 - e ** 2))
    thd = h / r ** 2
    rdot = MU / h * e * np.sin(nu)
    thdd = -2.0 * rdot * thd / r
    return r, thd, thdd


def true_relative_accel(t, x, v, e):
    """Exact (nonlinear) relative acceleration in the rotating LVLH frame about an
       eccentric orbit -- the GROUND TRUTH both plans are scored against. Handles
       2D (x = [radial, along-track]) and 3D (+ cross-track z); the cross-track
       term is the pure nonlinear gravity gradient -MU*z/rc^3."""
    r, thd, thdd = reference_kinematics(t, e)
    xr, ya = x[0], x[1]
    za = x[2] if len(x) == 3 else 0.0
    rc = np.sqrt((r + xr) ** 2 + ya ** 2 + za ** 2)
    ax = thdd * ya + 2 * thd * v[1] + thd ** 2 * xr - MU * (r + xr) / rc ** 3 + MU / r ** 2
    ay = -thdd * xr - 2 * thd * v[0] + thd ** 2 * ya - MU * ya / rc ** 3
    if len(x) == 3:
        az = -MU * za / rc ** 3
        return np.array([ax, ay, az])
    return np.array([ax, ay])


# ---------------------------------------------------------------------------
# 2. Plan in a chosen metric (CW: e=0 blocks; TH: the true eccentric blocks)
# ---------------------------------------------------------------------------
def plan(metric_e, start, goal, n, dt, w_bc=2.0e4):
    """Closed-form energy-optimal rest-to-rest plan in the metric of eccentricity
       ``metric_e`` (0 -> CW metric; e -> the correct time-varying TH metric).
       2D or 3D, inferred from len(start)."""
    dim = len(start)
    coeff = lambda tt: tschauner_hempel_coeffs(tt, metric_e, NM, a=A_REF, mu=MU, dim=dim)
    B, Minv, blocks = time_varying_operators(n, dt, coeff, w_bc=w_bc, dim=dim)
    c0 = time_varying_offset(blocks, start, goal)
    xi = -Minv @ (B.T @ c0)                      # unconstrained optimum = GN step
    return to_traj(xi, n)


# ---------------------------------------------------------------------------
# 3. Score a plan under the TRUE nonlinear eccentric dynamics (realized Delta-v)
# ---------------------------------------------------------------------------
def realized_delta_v(traj, start, goal, dt, e):
    """Delta-v (L1) a thruster must supply to realize ``traj`` under the true
       eccentric dynamics (2D or 3D): u_t = a_t^(plan) - f_true(x_t, v_t, t),
       summed |u|*dt. A plan that rides the true natural motion costs less."""
    Q = np.vstack([start, traj, goal])
    n = traj.shape[0]
    dv = 0.0
    for t in range(n):
        x = Q[t + 1]
        v = (Q[t + 2] - Q[t]) / (2.0 * dt)
        a = (Q[t + 2] - 2.0 * Q[t + 1] + Q[t]) / dt ** 2
        u = a - true_relative_accel((t + 1) * dt, x, v, e)
        dv += np.linalg.norm(u) * dt
    return dv


def passive_drift(e, rho=1.0e-3, orbits=1.0):
    """Diagnostic: how far a CW-passive 2:1 relative orbit drifts under the true
       eccentric dynamics in ``orbits`` revolutions, as a fraction of amplitude."""
    s0 = [rho, 0.0, 0.0, -2.0 * NM * rho]
    sol = solve_ivp(lambda t, s: [s[2], s[3],
                                  *true_relative_accel(t, s[:2], s[2:], e)],
                    [0, orbits * PERIOD], s0, max_step=PERIOD / 2000,
                    rtol=1e-10, atol=1e-12)
    drift = np.hypot(sol.y[0, -1] - s0[0], sol.y[1, -1] - s0[1])
    return drift / rho


# ---------------------------------------------------------------------------
# 4. Experiment
# ---------------------------------------------------------------------------
def run(eccentricities=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5), N=90,
        orbits=1.0, dim=2, verbose=True):
    """For each eccentricity, plan with the CW metric and with the TH metric, score
       both under the true (nonlinear) eccentric dynamics, and return the saving
       curve. ``dim=3`` adds an out-of-plane (cross-track) component to the
       transfer and scores against the full 3D nonlinear relative motion."""
    T = orbits * PERIOD
    dt = T / (N + 1)
    if dim == 2:
        start = np.array([0.08, 0.20])            # radial, along-track (normalized)
        goal = np.array([0.05, 0.0])
    else:
        start = np.array([0.08, 0.20, 0.06])      # + cross-track z
        goal = np.array([0.05, 0.0, 0.0])
    rows = []
    for e in eccentricities:
        t_cw = plan(0.0, start, goal, N, dt)      # circular-orbit metric (baseline)
        t_th = plan(e, start, goal, N, dt)        # correct time-varying metric
        dv_cw = realized_delta_v(t_cw, start, goal, dt, e)
        dv_th = realized_delta_v(t_th, start, goal, dt, e)
        saving = 100.0 * (dv_cw - dv_th) / dv_cw if dv_cw > 0 else 0.0
        rows.append(dict(e=e, dv_cw=dv_cw, dv_th=dv_th, saving=saving,
                         drift=passive_drift(e, orbits=orbits)))
    if verbose:
        print("=== COSMOS extension: time-varying metric for eccentric orbits ===")
        print(f"rest-to-rest relative transfer, {orbits:.0f} orbit(s), N={N} waypoints, "
              f"{dim}D ({'planar' if dim == 2 else 'with cross-track'})")
        print(f"ground truth: full nonlinear relative EOM about an eccentric orbit\n")
        print(f"{'e':>5} | {'CW-metric dv':>13} | {'TH-metric dv':>13} | "
              f"{'saving':>8} | {'CW-passive drift':>16}")
        print("-" * 70)
        for r in rows:
            print(f"{r['e']:5.2f} | {r['dv_cw']:13.5f} | {r['dv_th']:13.5f} | "
                  f"{r['saving']:7.1f}% | {r['drift']*100:14.0f}% amp")
        print("\nInterpretation: at e=0 the CW metric is exactly right (saving ~0,"
              " validating\nCW for circular LEO). As e grows the CW assumption breaks"
              " (a passive orbit\ndrifts many times its amplitude per revolution) and"
              " the correct time-varying\nmetric saves a large, real fraction of the"
              " Delta-v.")
    return rows


def main(out="cosmos_eccentric.png"):
    rows = run(verbose=True)
    es = [r["e"] for r in rows]
    savings = [r["saving"] for r in rows]
    drifts = [r["drift"] * 100 for r in rows]

    # also compute two example trajectories for the left panel (at a visible e)
    e_show = 0.3
    T = PERIOD; N = 90; dt = T / (N + 1)
    start = np.array([0.08, 0.20]); goal = np.array([0.05, 0.0])
    t_cw = plan(0.0, start, goal, N, dt)
    t_th = plan(e_show, start, goal, N, dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))
    Qcw = np.vstack([start, t_cw, goal]); Qth = np.vstack([start, t_th, goal])
    ax1.plot(Qcw[:, 1], Qcw[:, 0], "--", color="#888", lw=2.2,
             label="CW-metric plan (circular assumption)")
    ax1.plot(Qth[:, 1], Qth[:, 0], "-", color="#2563EB", lw=2.6,
             label=f"TH-metric plan (correct, e={e_show})")
    ax1.scatter(start[1], start[0], c="k", zorder=5, s=45, label="start")
    ax1.scatter(goal[1], goal[0], c="r", marker="*", s=160, zorder=5, label="target")
    ax1.set_xlabel("along-track  y"); ax1.set_ylabel("radial  x")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5)
    ax1.set_title(f"Same transfer, two metrics (eccentric reference, e={e_show})")

    # 3D (cross-track) sweep to show the generality holds out-of-plane
    rows3d = run(dim=3, verbose=False)
    es3 = [r["e"] for r in rows3d]; sav3 = [r["saving"] for r in rows3d]

    ax2.plot(es, savings, "o-", color="#1d9e75", lw=2.4,
             label="Δv saving — 2D (planar)")
    ax2.plot(es3, sav3, "^--", color="#2563EB", lw=2.0, alpha=0.9,
             label="Δv saving — 3D (with cross-track)")
    ax2.set_xlabel("reference-orbit eccentricity  e")
    ax2.set_ylabel("Δv saving  [%]")
    ax2.grid(alpha=0.3)
    ax2.set_title("When does the metric matter? (saving grows with eccentricity)")
    ax2.legend(fontsize=8.3, loc="upper left")

    fig.suptitle("COSMOS - time-varying metric for eccentric orbits "
                 "(Tschauner-Hempel): the generality claim, where it pays",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
