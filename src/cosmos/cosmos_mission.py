"""
COSMOS - A realistic SI mission: multi-chaser rendezvous on an eccentric orbit
==============================================================================

The other modules report normalized or dimensionless quantities. This one puts a
credible number on the table: a four-chaser debris rendezvous around a REAL
eccentric reference orbit, with the Delta-v reported in m/s, comparing the
COSMOS (correct time-varying / Tschauner-Hempel) metric to the dynamics-blind
(circular / Clohessy-Wiltshire) metric, scored under the full nonlinear relative
dynamics.

Why eccentric: cosmos_validity showed the fuel-aware metric pays in proportion to
the dynamics mismatch, which is governed by eccentricity. Real missions do fly
eccentric reference orbits (elliptical transfer/parking orbits, Molniya-class,
eccentric science orbits), and the formation amplitude (hundreds of metres)
is tiny next to the orbit (thousands of km), so the linearization is firmly valid
-- exactly the regime the validity map endorses.

SI scaling (normalized solver -> physical units)
------------------------------------------------
The time-varying operators are built in normalized units (semi-major axis a = 1,
mean motion n = 1), so a trajectory coordinate x_norm is in units of a and a
Delta-v dv_norm is in units of a*n. For a physical orbit (a, n) and a scene whose
coordinates are already expressed in units of a:

    distance_SI = a   * x_norm   [m]
    delta-v_SI  = a*n * dv_norm  [m/s]

We therefore express the formation geometry as a small fraction of a (a few
hundred metres over a multi-thousand-km orbit) and convert the result to m/s.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import to_traj
from ._core import (time_varying_operators, time_varying_offset,
                    tschauner_hempel_coeffs)

MU = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
RE = 6.371e6                 # Earth mean radius [m]


# ---------------------------------------------------------------------------
# normalized dynamics (a = 1, n = 1); eccentricity is the only shape parameter
# ---------------------------------------------------------------------------
def _refk(t, e):
    M = t; E = M
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    r = 1.0 - e * np.cos(E)
    h = np.sqrt(1.0 - e ** 2)
    thd = h / r ** 2
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    rdot = e * np.sin(nu) / h
    thdd = -2.0 * rdot * thd / r
    return r, thd, thdd


def _true_accel(t, x, v, e):
    r, thd, thdd = _refk(t, e)
    rc = np.sqrt((r + x[0]) ** 2 + x[1] ** 2)
    ax = thdd * x[1] + 2 * thd * v[1] + thd ** 2 * x[0] - (r + x[0]) / rc ** 3 + 1.0 / r ** 2
    ay = -thdd * x[0] - 2 * thd * v[0] + thd ** 2 * x[1] - x[1] / rc ** 3
    return np.array([ax, ay])


def _plan(metric_e, start, goal, N, dt, w_bc=2.0e4):
    coeff = lambda t: tschauner_hempel_coeffs(t, metric_e, 1.0, dim=2)
    B, Minv, blocks = time_varying_operators(N, dt, coeff, w_bc=w_bc, dim=2)
    c0 = time_varying_offset(blocks, start, goal)
    return to_traj(-Minv @ (B.T @ c0), N)


def _dv_norm(traj, start, goal, dt, e):
    Q = np.vstack([start, traj, goal]); dv = 0.0
    for t in range(traj.shape[0]):
        x = Q[t + 1]; v = (Q[t + 2] - Q[t]) / (2.0 * dt)
        a = (Q[t + 2] - 2.0 * Q[t + 1] + Q[t]) / dt ** 2
        dv += np.linalg.norm(a - _true_accel((t + 1) * dt, x, v, e)) * dt
    return dv


def orbit(perigee_km=600.0, apogee_km=8000.0):
    """Physical eccentric orbit from perigee/apogee altitudes -> (a, e, n, period)."""
    rp = RE + perigee_km * 1e3
    ra = RE + apogee_km * 1e3
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    n = np.sqrt(MU / a ** 3)
    return dict(a=a, e=e, n=n, period=2 * np.pi / n, rp=rp, ra=ra)


def scenario(a, scene_m=200.0):
    """4 chasers (a launch cluster) -> 4 debris slots on a relative ring. Geometry
       expressed in units of a (so coordinates are ~ scene_m / a)."""
    s = scene_m / a
    starts = [np.array([s * x, -3 * s]) for x in (-1.5, -0.5, 0.5, 1.5)]
    goals = [np.array([2 * s * np.cos(th), 2 * s * np.sin(th)])
             for th in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    return starts, goals


def run(perigee_km=600.0, apogee_km=8000.0, scene_m=200.0, N=90, verbose=True):
    o = orbit(perigee_km, apogee_km)
    a, e, n = o["a"], o["e"], o["n"]
    starts, goals = scenario(a, scene_m)
    dt = np.pi / (N + 1)                       # half-orbit maneuver (normalized T=pi)
    K_DV = a * n                               # normalized dv -> m/s

    cos, blind, dvc, dvb = [], [], 0.0, 0.0
    for s, g in zip(starts, goals):
        tc = _plan(e, s, g, N, dt); tb = _plan(0.0, s, g, N, dt)
        cos.append(tc); blind.append(tb)
        dvc += _dv_norm(tc, s, g, dt, e); dvb += _dv_norm(tb, s, g, dt, e)
    dvc_si, dvb_si = dvc * K_DV, dvb * K_DV
    K = len(starts)
    if verbose:
        print("=== COSMOS mission: multi-chaser rendezvous on a real eccentric orbit ===")
        print(f"reference orbit : perigee {perigee_km:.0f} km, apogee {apogee_km:.0f} km")
        print(f"                  a = {a/1e3:.0f} km, e = {e:.3f}, period = {o['period']/60:.0f} min")
        print(f"formation scale : {scene_m:.0f} m  (amplitude/a = {scene_m/a:.1e}, linear regime)")
        print(f"maneuver        : half an orbit ({o['period']/120:.0f} min), {K} chasers\n")
        print(f"COSMOS (eccentric metric): total Δv {dvc_si:.3f} m/s   "
              f"({dvc_si/K:.3f} m/s per chaser)")
        print(f"blind  (circular metric) : total Δv {dvb_si:.3f} m/s   "
              f"({dvb_si/K:.3f} m/s per chaser)")
        print(f"saving                   : {100*(dvb_si-dvc_si)/dvb_si:+.1f} %  "
              f"({dvb_si-dvc_si:.3f} m/s of propellant, scored under true nonlinear motion)")
    return dict(orbit=o, starts=starts, goals=goals, cos=cos, blind=blind,
                dvc=dvc_si, dvb=dvb_si, scene_m=scene_m, K_X=a)


def main(out="cosmos_mission.png"):
    r = run(verbose=True)
    o = r["orbit"]; starts, goals = r["starts"], r["goals"]
    a = r["K_X"]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, len(starts)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # left: the SI trajectories (metres)
    for k in range(len(starts)):
        Qc = np.vstack([starts[k], r["cos"][k], goals[k]]) * a
        Qb = np.vstack([starts[k], r["blind"][k], goals[k]]) * a
        ax1.plot(Qb[:, 1], Qb[:, 0], "--", color=colors[k], lw=1.1, alpha=0.5)
        ax1.plot(Qc[:, 1], Qc[:, 0], "-", color=colors[k], lw=2.4)
        ax1.scatter(starts[k][1] * a, starts[k][0] * a, color=colors[k], s=42, zorder=5)
        ax1.scatter(goals[k][1] * a, goals[k][0] * a, color=colors[k], marker="*",
                    s=150, zorder=5)
    ax1.plot([], [], "--", color="#888", label="blind (circular metric)")
    ax1.plot([], [], "-", color="#333", label="COSMOS (eccentric metric)")
    ax1.set_xlabel("along-track  [m]"); ax1.set_ylabel("radial  [m]")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    ax1.set_title(f"4-chaser rendezvous (e = {o['e']:.2f}, "
                  f"perigee {int((o['rp']-RE)/1e3)} / apogee {int((o['ra']-RE)/1e3)} km)")

    # right: Delta-v bars in m/s
    vals = [r["dvb"], r["dvc"]]; labs = ["blind\n(circular)", "COSMOS\n(eccentric)"]
    cols = ["#d96459", "#1d9e75"]
    ax2.bar(labs, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.05, f"{v:.2f} m/s", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("total Δv  [m/s]")
    ax2.set_ylim(0, max(vals) * 1.22)
    ax2.set_title(f"Propellant: {100*(r['dvb']-r['dvc'])/r['dvb']:+.0f}% "
                  f"with the correct metric")
    fig.suptitle("COSMOS - a realistic SI mission: credible m/s on an eccentric orbit",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
