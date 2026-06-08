"""
COSMOS - Orbital real-time simulation (visual demo)
===================================================

Multi-agent CHOMP put into an orbital scene: Earth at the center is a KEEP-OUT
zone (a static obstacle), and K satellites must reach their target on the other
side, going AROUND the Earth and AVOIDING each other.

This is exactly F_obs (Earth) + F_agents (inter-satellite) from Part II, with
the trajectories then ANIMATED so you watch the satellites move in real time.

Note: this uses free-space kinematics for the visual. Real orbital dynamics
(Clohessy-Wiltshire in the Hill frame) is Part III of the framework.

Run:
  python3 orbital_sim.py          # saves orbital_sim.gif
  set SHOW = True (below) + a GUI backend for a live real-time window.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .chomp_swarm import solve_swarm

SHOW = False   # True -> live window (needs a GUI backend instead of Agg)


# ----------------------------------------------------------------------------
# 1. Scene setup: Earth + satellites on a ring, targets antipodal
# ----------------------------------------------------------------------------
K = 5
R_earth = 2.5
R_ring  = 7.0
n_wp    = 60

ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
starts = [np.array([R_ring * np.cos(a), R_ring * np.sin(a)]) for a in ang]
# target = antipodal point on the ring -> straight line would cross the Earth
goals  = [np.array([R_ring * np.cos(a + np.pi), R_ring * np.sin(a + np.pi)])
          for a in ang]

earth = [{"c": np.array([0.0, 0.0]), "r": R_earth}]

# ----------------------------------------------------------------------------
# 2. Plan collision-free trajectories (Gauss-Seidel = best from Part II)
# ----------------------------------------------------------------------------
print("Planning trajectories (Gauss-Seidel)...")
trajs, hist, _ = solve_swarm(
    starts, goals, obstacles=earth, strategy="gauss_seidel",
    n=n_wp, dt=1.0, lam=0.6, mu=1.2, eps_o=0.9, eps_a=1.0,
    r_safe=0.9, step=0.02, iters=400, seed=3)

paths = [np.vstack([starts[k], trajs[k], goals[k]]) for k in range(K)]  # (n+2,2)

# quick safety report
def earth_clearance(paths):
    return min(np.min(np.linalg.norm(p, axis=1)) - R_earth for p in paths)
def inter_dist(paths):
    d = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = min(d, np.min(np.linalg.norm(paths[i] - paths[j], axis=1)))
    return d
print(f"min Earth clearance   : {earth_clearance(paths):+.3f}")
print(f"min inter-sat distance: {inter_dist(paths):+.3f}")

# ----------------------------------------------------------------------------
# 3. Animation
# ----------------------------------------------------------------------------
N_FRAMES = 90
TRAIL = 18
colors = plt.cm.turbo(np.linspace(0.12, 0.92, K))

fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.set_facecolor("#05060a")
fig.patch.set_facecolor("#05060a")
ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# Earth
th = np.linspace(0, 2 * np.pi, 200)
ax.fill(R_earth * np.cos(th), R_earth * np.sin(th), color="#1f6feb", zorder=2)
ax.plot((R_earth + 0.15) * np.cos(th), (R_earth + 0.15) * np.sin(th),
        color="#3a8bff", lw=1, alpha=0.4, zorder=2)
ax.text(0, 0, "Earth", color="white", ha="center", va="center",
        fontsize=11, zorder=3)

# faint planned paths + targets
for k in range(K):
    ax.plot(paths[k][:, 0], paths[k][:, 1], color=colors[k], lw=0.8,
            alpha=0.25, zorder=1)
    ax.scatter(*goals[k], color=colors[k], s=60, marker="x", alpha=0.7, zorder=3)

trails = [ax.plot([], [], color=colors[k], lw=2.2, alpha=0.9, zorder=4)[0]
          for k in range(K)]
dots = [ax.scatter([], [], color=colors[k], s=70, edgecolor="white",
                   linewidth=0.8, zorder=5) for k in range(K)]
title = ax.text(0, 8.4, "", color="white", ha="center", fontsize=12)


def sample(path, s):
    """Position along a path at fractional progress s in [0, 1]."""
    idx = s * (len(path) - 1)
    i0 = int(np.floor(idx)); i0 = min(i0, len(path) - 2)
    f = idx - i0
    return (1 - f) * path[i0] + f * path[i0 + 1]


def init():
    for tr in trails:
        tr.set_data([], [])
    return trails + dots + [title]


def update(frame):
    s = frame / (N_FRAMES - 1)
    title.set_text(f"COSMOS - orbital transfer   t = {s*100:4.0f}%")
    for k in range(K):
        pos = sample(paths[k], s)
        dots[k].set_offsets([pos])
        lo = max(0.0, s - TRAIL / (len(paths[k])))
        ss = np.linspace(lo, s, TRAIL)
        pts = np.array([sample(paths[k], u) for u in ss])
        trails[k].set_data(pts[:, 0], pts[:, 1])
    return trails + dots + [title]


anim = FuncAnimation(fig, update, frames=N_FRAMES, init_func=init,
                     blit=True, interval=50)

if SHOW:
    plt.show()
else:
    anim.save("orbital_sim.gif", writer=PillowWriter(fps=20), dpi=80)
    print("Animation -> orbital_sim.gif")
