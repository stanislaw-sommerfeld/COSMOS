"""
COSMOS - Why the dynamics matter: CW vs J2 (illustration / future-work motivation)
==================================================================================

This is NOT a planner and NOT part of the mission. It is a one-figure argument for
the "metric = dynamics" point: the SAME passive relative orbit that is *closed and
fuel-free* under Clohessy-Wiltshire DRIFTS under the J2 oblateness perturbation, so
"free" coast arcs are not actually free once J2 is present.

We forward-propagate one initial condition (a passive CW 2:1 relative orbit) under
two LINEAR models with identical structure, differing only by the coefficient block G:

  CW  : x'' - 2 n y'  - 3 n^2 x       = 0 ;  y'' + 2 n x'  = 0
  SS  : x'' - 2 n c y' - (5c^2-2)n^2 x = 0 ;  y'' + 2 n c x' = 0       (Schweighart-Sedwick)
        with c = sqrt(1 + s),  s = (3 J2 Re^2 / 8 a^2)(1 + 3 cos 2i)

c -> 1 (J2 -> 0) recovers CW exactly: the figure's left panel collapses onto the right.
This is the same "B = D2 - G, swap G" idea behind COSMOS, shown at the dynamics level.

Reference: Schweighart & Sedwick, "High-Fidelity Linearized J2 Model for Satellite
Formation Flight", J. Guidance, Control, and Dynamics, 2002.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- physical constants (LEO, 500 km) --------------------------------------
MU = 3.986004418e14            # Earth GM [m^3/s^2]
RE = 6.378137e6                # Earth equatorial radius [m]
J2 = 1.08263e-3                # Earth oblateness coefficient
ALT = 500e3                    # altitude [m]
INC = np.deg2rad(51.6)         # inclination (ISS-like) [rad]

a = RE + ALT                   # reference semi-major axis [m]
n = np.sqrt(MU / a ** 3)       # mean motion [rad/s]
period = 2 * np.pi / n         # orbital period [s]

# Schweighart-Sedwick J2 parameter and corrected rate
s = (3.0 * J2 * RE ** 2) / (8.0 * a ** 2) * (1.0 + 3.0 * np.cos(2.0 * INC))
c = np.sqrt(1.0 + s)


def A_cw(nm):
    return np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                     [3 * nm ** 2, 0, 0, 2 * nm], [0, 0, -2 * nm, 0]], float)


def A_ss(nm, cc):
    """Same structure as CW; only the coefficient block G changes (the whole point)."""
    return np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                     [(5 * cc ** 2 - 2) * nm ** 2, 0, 0, 2 * nm * cc],
                     [0, 0, -2 * nm * cc, 0]], float)


def propagate(A, s0, T, npts):
    """Exact linear propagation of [x, y, vx, vy] via the state-transition matrix."""
    h = T / (npts - 1)
    Phi = expm(A * h)
    out = np.zeros((npts, 4))
    out[0] = s0
    s = s0.copy()
    for k in range(1, npts):
        s = Phi @ s
        out[k] = s
    return out


def main():
    rho = 100.0                       # radial amplitude of the relative orbit [m]
    # passive CW 2:1 ellipse: x0=rho, y0=0, vx0=0, vy0=-2 n rho  (no CW along-track drift)
    s0 = np.array([rho, 0.0, 0.0, -2.0 * n * rho])
    n_orbits = 180                    # ~12.6 days: long enough for the REAL drift to show
    T = n_orbits * period
    npts = n_orbits * 30

    cw = propagate(A_cw(n), s0, T, npts)
    ss = propagate(A_ss(n, c), s0, T, npts)            # REALISTIC J2 — no exaggeration
    t_days = np.linspace(0, T, npts) / 86400.0
    sep = np.linalg.norm(cw[:, :2] - ss[:, :2], axis=1)
    drift_per_orbit = (ss[-1, 1] - cw[-1, 1]) / n_orbits
    drift_per_day = drift_per_orbit * (86400.0 / period)

    print("=== CW vs J2 (Schweighart-Sedwick) — REALISTIC, no exaggeration ===")
    print(f"altitude {ALT/1e3:.0f} km | inclination {np.rad2deg(INC):.1f} deg | "
          f"period {period/60:.1f} min")
    print(f"J2 parameter s = {s:.3e}  ->  c = sqrt(1+s) = {c:.8f}  (c-1 = {c-1:.2e})")
    print(f"relative-orbit amplitude rho = {rho:.0f} m, propagated {n_orbits} orbits "
          f"(~{T/86400:.1f} days)")
    print(f"secular along-track drift: {drift_per_orbit:+.3f} m/orbit "
          f"(~{drift_per_day:+.2f} m/day); reaches {sep[-1]:.1f} m after {n_orbits} orbits.")
    print("This is the TRUE J2 magnitude: small per orbit, but secular -> it accumulates")
    print("and must be cancelled continuously, so a CW-'passive' orbit is not J2-passive.")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))

    # panel 1: in-plane orbit. CW = one closed ellipse (it retraces). J2 = time-coloured
    # so the slow drift of the SAME orbit is visible at its real magnitude.
    one = int(npts / n_orbits) + 1
    ax1.plot(cw[:one, 1], cw[:one, 0], color="#2563EB", lw=2.2,
             label="Clohessy–Wiltshire (closed — retraces every orbit)")
    pts = ax1.scatter(ss[:, 1], ss[:, 0], c=t_days, cmap="autumn_r", s=2, alpha=0.7)
    ax1.scatter([s0[1]], [s0[0]], c="k", zorder=5, s=30, label="start")
    cb = fig.colorbar(pts, ax=ax1, fraction=0.046, pad=0.04); cb.set_label("days", fontsize=9)
    ax1.set_xlabel("along-track  y  [m]"); ax1.set_ylabel("radial  x  [m]")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8.2, loc="upper right")
    ax1.set_title(f"Same passive orbit under J2, real magnitude ({n_orbits} orbits)")

    # panel 2: secular growth of the CW–J2 separation. The instantaneous error
    # oscillates within each orbit, so we draw it faintly and overlay the secular
    # ENVELOPE (running max) as the clean line that grows ~linearly with time.
    env = np.maximum.accumulate(sep)
    ax2.plot(t_days, sep, color="#DC2626", lw=0.5, alpha=0.30,
             label="instantaneous error (oscillates each orbit)")
    ax2.plot(t_days, env, color="#DC2626", lw=2.6,
             label=r"secular envelope (grows $\propto t$)")
    ax2.fill_between(t_days, 0, env, color="#DC2626", alpha=0.06)
    ax2.set_xlabel("time  [days]"); ax2.set_ylabel(r"CW $\to$ J2 position error  [m]")
    ax2.grid(alpha=0.3); ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_title("Secular drift: tiny per orbit, but unbounded over time")
    ax2.annotate(f"≈ {abs(drift_per_day):.1f} m/day\nfor a {rho:.0f} m formation",
                 xy=(t_days[-1], env[-1]), xytext=(t_days[-1] * 0.42, env[-1] * 0.66),
                 fontsize=10, color="#7a2a1a",
                 arrowprops=dict(arrowstyle="->", color="#7a2a1a", lw=1.2))

    fig.suptitle("COSMOS — why the dynamics matter: a CW-passive orbit is not J2-passive "
                 "(realistic J2 at 500 km)", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "No exaggeration: this is the true Schweighart–Sedwick J2 effect. The drift is "
             "tiny per orbit but secular, so 'free' coast is not free under the true dynamics "
             "— feed the planner the right G and 'free' means free for real.",
             ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("cw_vs_j2.png", dpi=130)
    print("\nFigure -> cw_vs_j2.png")


if __name__ == "__main__":
    main()
