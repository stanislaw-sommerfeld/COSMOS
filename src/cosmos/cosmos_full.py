"""
COSMOS - Full system: multi-agent fuel-aware rendezvous in the Hill frame
=========================================================================

This integrates the three parts:
  Part II (multi-agent, Gauss-Seidel decoupling, inter-agent avoidance)
  Part III (fuel-aware Clohessy-Wiltshire metric  A^T A -> B^T B)
plus a RENDEZVOUS boundary condition: each chaser must arrive at its assigned
debris with (near) zero relative velocity -- and start at rest in formation.

Mission modelled: the RENDEZVOUS phase of debris removal. K chasers leave a
formation (at rest) and each reaches its assigned debris target with matched
velocity (soft rendezvous). Capture + deorbit are downstream and abstracted.

Frame: LOCAL Hill frame (x = radial, y = along-track). The Earth is NOT here
(it is ~thousands of km away in -x); the central keep-out is a structure/object.

Reuses: cw_operators / cw_offset / to_flat / to_traj (Part III),
        repulsion_grad (Part II).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import cw_operators, cw_offset, to_flat, to_traj, cw_control
from .chomp_swarm import repulsion_grad


# ============================================================================
# 1. One covariant step for an agent, in the fuel metric, with rendezvous BC
# ============================================================================
def step_agent_cw(traj, start, goal, B, Minv, c0, dt,
                  static, agents, lam, mu, eps_o, eps_a, step):
    n = traj.shape[0]
    xi = to_flat(traj)
    U = B @ xi + c0                                       # includes BC rows
    g = lam * (B.T @ U)                                   # fuel + rendezvous BC

    if static:
        g_o, _ = repulsion_grad(traj, start, goal, dt, static, eps_o)
        g = g + to_flat(g_o)
    if agents:
        g_a, _ = repulsion_grad(traj, start, goal, dt, agents, eps_a)
        g = g + mu * to_flat(g_a)

    xi = xi - step * (Minv @ g)
    return to_traj(xi, n)


def project_keepout(traj, obstacles, margin=0.05):
    """Hard safety (metric-BLIND, Euclidean): push any waypoint inside a keep-out
       straight out to its boundary. Kept for comparison / ablation."""
    for i in range(len(traj)):
        for o in obstacles:
            v = traj[i] - o["c"]
            dist = np.linalg.norm(v)
            if dist - o["r"] < margin:
                traj[i] = o["c"] + v / (dist + 1e-9) * (o["r"] + margin)
    return traj


def project_keepout_metric(traj, obstacles, Minv, margin=0.05, sweeps=3):
    """Metric-AWARE (Riemannian) keep-out projection in M = B^T B, dim-general.
       Same closed form as Module B (cosmos_safe.project_metric), generalized to
       2D/3D. This is what the mission driver (cli.py) uses; cosmos_full now matches
       it so the 'full system' demo does not silently fall back to the 15x-blind
       projection that Module A/B identified and fixed."""
    n, D = traj.shape
    xi = to_flat(traj)
    for _ in range(sweeps):
        Tj = to_traj(xi, n)
        for i in range(n):
            for o in obstacles:
                v = Tj[i] - o["c"]
                dist = np.linalg.norm(v) + 1e-9
                clr = dist - o["r"]
                if clr < margin:
                    ni = v / dist
                    a = np.zeros(D * n)
                    for dd in range(D):
                        a[dd * n + i] = ni[dd]
                    Ma = Minv @ a
                    xi = xi + Ma * ((margin - clr) / (a @ Ma))
                    Tj = to_traj(xi, n)
    return to_traj(xi, n)


def project_inter_agent(trajs, r_col):
    """Hard safety: at each time index, push apart any pair closer than r_col."""
    K = len(trajs); n = trajs[0].shape[0]
    for t in range(n):
        for i in range(K):
            for j in range(i + 1, K):
                d = trajs[i][t] - trajs[j][t]
                dist = np.linalg.norm(d)
                if dist < r_col:
                    push = 0.5 * (r_col - dist) * d / (dist + 1e-9)
                    trajs[i][t] = trajs[i][t] + push
                    trajs[j][t] = trajs[j][t] - push
    return trajs


# ============================================================================
# 2. Multi-agent solver (Gauss-Seidel) in the fuel metric
# ============================================================================
def solve_swarm_cw(starts, goals, obstacles=(), nm=1.0, n=60, dt=1.0,
                   lam=1.0, mu=1.5, eps_o=0.9, eps_a=1.0, r_safe=0.9,
                   w_bc=2.0, step=0.02, iters=500, seed=0, r_col=0.0,
                   safety="riemann"):
    K = len(starts)
    D = len(starts[0])                                   # 2 or 3
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc, dim=D)
    c0s = [cw_offset(blocks, starts[k], goals[k]) for k in range(K)]
    static = [{"pos": np.tile(o["c"], (n + 2, 1)), "rad": o["r"]} for o in obstacles]

    rng = np.random.default_rng(seed)
    trajs = [np.linspace(starts[k], goals[k], n + 2)[1:-1]
             + rng.normal(0, 0.05, (n, D)) for k in range(K)]

    for _ in range(iters):
        for k in range(K):                                # Gauss-Seidel
            agents = [{"pos": np.vstack([starts[j], trajs[j], goals[j]]),
                       "rad": r_safe} for j in range(K) if j != k]
            trajs[k] = step_agent_cw(trajs[k], starts[k], goals[k], B, Minv,
                                     c0s[k], dt, static, agents, lam, mu,
                                     eps_o, eps_a, step)
            # rendezvous: soft terminal-velocity penalty lives in the metric
            # (cw_operators w_bc). Here we enforce keep-out (hard safety), in the
            # B^T B metric by default (matches the mission driver and Module B).
            if obstacles:
                if safety == "euclid":
                    trajs[k] = project_keepout(trajs[k], obstacles)
                else:
                    trajs[k] = project_keepout_metric(trajs[k], obstacles, Minv)
        if r_col > 0.0:                                   # hard inter-agent safety
            trajs = project_inter_agent(trajs, r_col)
    return trajs


# ============================================================================
# 3. Metrics
# ============================================================================
def total_delta_v(trajs, starts, goals, dt, nm):
    dv = 0.0
    for k in range(len(trajs)):
        U = cw_control(trajs[k], starts[k], goals[k], dt, nm)
        dv += np.sum(np.linalg.norm(U, axis=1)) * dt
    return dv

def arrival_speed(trajs, goals, dt):
    """Relative speed at the moment of arrival (should be ~0 for rendezvous)."""
    sp = []
    for k in range(len(trajs)):
        v = (goals[k] - trajs[k][-1]) / dt   # terminal velocity
        sp.append(np.linalg.norm(v))
    return np.array(sp)

def min_inter_agent(trajs, starts, goals):
    K = len(trajs)
    Qs = [np.vstack([starts[k], trajs[k], goals[k]]) for k in range(K)]
    d = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = min(d, np.min(np.linalg.norm(Qs[i] - Qs[j], axis=1)))
    return d

def min_keepout_clearance(trajs, starts, goals, obstacles):
    if not obstacles:
        return np.nan
    d = np.inf
    for k in range(len(trajs)):
        Q = np.vstack([starts[k], trajs[k], goals[k]])
        for o in obstacles:
            d = min(d, np.min(np.linalg.norm(Q - o["c"], axis=1)) - o["r"])
    return d


# ============================================================================
# 4. SCENARIO: 4 chasers rendezvous with 4 debris targets, around a keep-out
# ============================================================================
if __name__ == "__main__":
    import sys
    DIM = 3 if any(a in ("3", "3d", "--3d") for a in sys.argv[1:]) else 2

    nm = 1.0
    T = np.pi
    N = 60
    dt = T / (N + 1)

    if DIM == 2:
        starts = [np.array([x, -6.0]) for x in (-1.5, -0.5, 0.5, 1.5)]
        goals = [np.array(g) for g in [(-4.0, 3.0), (4.0, 3.0),
                                       (-3.0, -2.0), (3.0, -2.0)]]
    else:   # 3D: out-of-plane debris targets (cross-track z)
        starts = [np.array([x, -6.0, 0.0]) for x in (-1.5, -0.5, 0.5, 1.5)]
        goals = [np.array(g) for g in [(-4.0, 3.0, 2.5), (4.0, 3.0, -2.5),
                                       (-3.0, -2.0, 3.0), (3.0, -2.0, -3.0)]]
    obstacles = [{"c": np.zeros(DIM), "r": 1.6}]

    common = dict(n=N, dt=dt, lam=1.0, mu=2.0, eps_o=0.8, eps_a=1.0,
                  r_safe=0.9, w_bc=300.0, step=0.02, iters=800, seed=2, r_col=0.6)

    trajs_fuel = solve_swarm_cw(starts, goals, obstacles, nm=nm, **common)
    trajs_free = solve_swarm_cw(starts, goals, obstacles, nm=0.0, **common)

    dv_fuel = total_delta_v(trajs_fuel, starts, goals, dt, nm)
    dv_free = total_delta_v(trajs_free, starts, goals, dt, nm)
    av = arrival_speed(trajs_fuel, goals, dt)

    print(f"=== COSMOS full system ({DIM}D): multi-agent fuel-aware rendezvous ===")
    print(f"total delta-v  fuel-aware : {dv_fuel:.3f}")
    print(f"total delta-v  free-space : {dv_free:.3f}   "
          f"(fuel-aware saves {100*(dv_free-dv_fuel)/dv_free:+.0f}%)")
    print(f"min inter-agent distance  : {min_inter_agent(trajs_fuel, starts, goals):+.3f}")
    print(f"min keep-out clearance    : {min_keepout_clearance(trajs_fuel, starts, goals, obstacles):+.3f}")
    print(f"arrival speeds (rdv ~0)   : {np.array2string(av, precision=3)}")

    colors = plt.cm.turbo(np.linspace(0.12, 0.92, len(starts)))

    if DIM == 2:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        th = np.linspace(0, 2 * np.pi, 160)
        for o in obstacles:
            ax.fill(o["c"][1] + o["r"] * np.cos(th), o["c"][0] + o["r"] * np.sin(th),
                    color="#d96459", alpha=0.30)
            ax.text(o["c"][1], o["c"][0], "keep-out", ha="center", va="center",
                    color="#7a2a1a", fontsize=9)
        for k in range(len(starts)):
            Qu = np.vstack([starts[k], trajs_fuel[k], goals[k]])
            Qf = np.vstack([starts[k], trajs_free[k], goals[k]])
            ax.plot(Qf[:, 1], Qf[:, 0], "--", color=colors[k], lw=1, alpha=0.4)
            ax.plot(Qu[:, 1], Qu[:, 0], "-", color=colors[k], lw=2.4)
            ax.scatter(starts[k][1], starts[k][0], color=colors[k], s=45, zorder=5)
            ax.scatter(goals[k][1], goals[k][0], color=colors[k], s=150,
                       marker="*", zorder=5)
        ax.set_xlabel("along-track  y"); ax.set_ylabel("radial  x")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        out = "cosmos_full.png"
    else:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        u = np.linspace(0, 2 * np.pi, 20); v = np.linspace(0, np.pi, 12)
        c, r = obstacles[0]["c"], obstacles[0]["r"]
        sx = c[0] + r * np.outer(np.cos(u), np.sin(v))
        sy = c[1] + r * np.outer(np.sin(u), np.sin(v))
        sz = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color="#d96459", alpha=0.35, linewidth=0.5)
        for k in range(len(starts)):
            Qu = np.vstack([starts[k], trajs_fuel[k], goals[k]])
            ax.plot(Qu[:, 0], Qu[:, 1], Qu[:, 2], "-", color=colors[k], lw=2.2)
            ax.scatter(*starts[k], color=colors[k], s=40)
            ax.scatter(*goals[k], color=colors[k], s=130, marker="*")
        ax.set_xlabel("radial x"); ax.set_ylabel("along-track y")
        ax.set_zlabel("cross-track z")
        out = "cosmos_full_3d.png"

    ax.set_title(f"COSMOS - multi-agent fuel-aware rendezvous ({DIM}D, Hill frame)\n"
                 f"total delta-v: fuel-aware {dv_fuel:.2f} vs free-space {dv_free:.2f}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nFigure -> {out}")
