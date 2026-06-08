"""
COSMOS - Realistic LEO units (SI: meters, m/s)
==============================================

Our normalized solver (nm=1, T=pi) IS the non-dimensional LEO case under the
scaling  t' = n*t,  x' = x/L. We therefore pick a real LEO orbit + a length
scale L, run the existing solver, and convert outputs to SI:

    distance_SI  = L  * x_norm          [m]
    velocity_SI  = L*n * v_norm          [m/s]
    delta-v_SI   = L*n * dv_norm         [m/s]   (since dv ~ velocity)

Derivation of the velocity/dv factor: x = L x', t = t'/n  =>  v = dx/dt = L n v',
and u = L n^2 u', so delta-v = int||u|| dt = L n int||u'|| dt' = L n dv'.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .cosmos_full import solve_swarm_cw, total_delta_v, arrival_speed, cw_control
from .chomp_cw import cw_control as _cwc  # per-agent control

# ---------------------------------------------------------------------------
# 1. Real LEO orbit
# ---------------------------------------------------------------------------
MU = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
RE = 6.371e6                 # Earth radius [m]
ALT = 500e3                  # orbit altitude [m] (typical LEO)
a = RE + ALT                 # orbit semi-major axis [m]
n = np.sqrt(MU / a ** 3)     # mean motion [rad/s]
T_orb = 2 * np.pi / n        # orbital period [s]

# ---------------------------------------------------------------------------
# 2. Scaling: length scale L, half-orbit maneuver (matches normalized T=pi)
# ---------------------------------------------------------------------------
L = 100.0                    # length scale [m]  -> normalized "1" = 100 m
N = 50
dt_norm = np.pi / (N + 1)    # normalized step (T_norm = pi  ==  half orbit)
T_man = np.pi / n            # maneuver duration in SI [s]
K_DV = L * n                 # norm -> m/s  (velocity & delta-v)
K_X = L                      # norm -> m    (distance)

# ---------------------------------------------------------------------------
# 3. Scenario (deterministic, in NORMALIZED units) - 4 chasers, 4 debris
# ---------------------------------------------------------------------------
starts = [np.array([x, -6.0]) for x in (-1.5, -0.5, 0.5, 1.5)]
goals = [np.array(g) for g in [(-4.0, 3.0), (4.0, 3.0), (-3.0, -2.0), (3.0, -2.0)]]
obstacles = [{"c": np.array([0.0, 0.0]), "r": 1.6}]

common = dict(n=N, dt=dt_norm, lam=1.0, mu=3.0, eps_o=0.8, eps_a=1.2,
              r_safe=0.9, w_bc=300.0, step=0.02, iters=800, seed=2, r_col=0.0)

trajs_fuel = solve_swarm_cw(starts, goals, obstacles, nm=1.0, **common)
trajs_free = solve_swarm_cw(starts, goals, obstacles, nm=0.0, **common)

# ---------------------------------------------------------------------------
# 4. Convert to SI and report
# ---------------------------------------------------------------------------
dv_fuel = total_delta_v(trajs_fuel, starts, goals, dt_norm, 1.0) * K_DV
dv_free = total_delta_v(trajs_free, starts, goals, dt_norm, 1.0) * K_DV
per_chaser = [ _cwc(trajs_fuel[k], starts[k], goals[k], dt_norm, 1.0) for k in range(4)]
per_dv = [np.sum(np.linalg.norm(U, axis=1)) * dt_norm * K_DV for U in per_chaser]
arr = arrival_speed(trajs_fuel, goals, dt_norm) * K_DV

print("=== COSMOS in realistic LEO units ===")
print(f"orbit altitude        : {ALT/1e3:.0f} km")
print(f"mean motion n         : {n:.3e} rad/s")
print(f"orbital period        : {T_orb/60:.1f} min")
print(f"maneuver time (half)  : {T_man/60:.1f} min")
print(f"length scale L        : {L:.0f} m  (scene spans ~+/- {6*L:.0f} m,"
      f" keep-out radius {1.6*L:.0f} m)")
print("-" * 48)
print(f"total delta-v  fuel-aware : {dv_fuel:.3f} m/s")
print(f"total delta-v  free-space : {dv_free:.3f} m/s")
print(f"per-chaser fuel-aware dv  : "
      f"{', '.join(f'{d:.3f}' for d in per_dv)} m/s")
print(f"arrival speeds (rendezvous): "
      f"{', '.join(f'{v:.3f}' for v in arr)} m/s")
print(f"saving                    : {100*(dv_free-dv_fuel)/dv_free:+.0f}%")

# ---------------------------------------------------------------------------
# 5. Figure in SI (meters), delta-v in m/s
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 7.2))
colors = plt.cm.turbo(np.linspace(0.12, 0.92, 4))
th = np.linspace(0, 2 * np.pi, 160)
for o in obstacles:
    ax.fill((o["c"][1] + o["r"] * np.cos(th)) * L,
            (o["c"][0] + o["r"] * np.sin(th)) * L, color="#d96459", alpha=0.30)
    ax.text(o["c"][1] * L, o["c"][0] * L, "keep-out", ha="center", va="center",
            color="#7a2a1a", fontsize=9)
for k in range(4):
    Qu = np.vstack([starts[k], trajs_fuel[k], goals[k]]) * L
    ax.plot(Qu[:, 1], Qu[:, 0], "-", color=colors[k], lw=2.4)
    ax.scatter(starts[k][1] * L, starts[k][0] * L, color=colors[k], s=45, zorder=5)
    ax.scatter(goals[k][1] * L, goals[k][0] * L, color=colors[k], s=150,
               marker="*", zorder=5)
ax.scatter([], [], color="k", s=45, label="chaser start (at rest)")
ax.scatter([], [], color="k", s=150, marker="*", label="debris target")
ax.set_xlabel("along-track  y  [m]"); ax.set_ylabel("radial  x  [m]")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
ax.set_title(f"COSMOS - multi-chaser debris rendezvous, LEO {ALT/1e3:.0f} km\n"
             f"total delta-v: fuel-aware {dv_fuel:.2f} m/s  vs  "
             f"free-space {dv_free:.2f} m/s   ({100*(dv_free-dv_fuel)/dv_free:.0f}% saved)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("cosmos_units.png", dpi=120)
print("\nFigure -> cosmos_units.png")
