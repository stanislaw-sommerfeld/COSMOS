"""
COSMOS - CHOMP multi-agent / "Swarm" (Framework Part II)
========================================================

We extend Part I: each agent optimizes ITS trajectory, but sees the OTHER
agents as "moving obstacles". The key pedagogical point:

    A STATIC OBSTACLE and ANOTHER AGENT are EXACTLY the same repulsion
    computation. The only difference: the "source" position is constant
    (obstacle) or time-varying (agent). -> a single function handles both.

We compare 3 ways to handle the coupling xi_i <-> xi_j (the "decoupling"):

  - JACOBI        : all updated in parallel, xi_j frozen at iteration k.
  - GAUSS-SEIDEL  : sequential, the already-updated xi_j is used immediately.
  - PRIORITY      : fixed order; each agent avoids the FINAL trajectories of
                    higher-priority agents (collision-free by construction).

Reuses build_smoothness / c_cost / c_grad from Part I.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_single_agent import build_smoothness, c_cost, c_grad


# ============================================================================
# 1. UNIFIED REPULSION  (static obstacles AND moving agents = same code)
# ============================================================================
def repulsion_grad(traj, start, goal, dt, sources, eps):
    """
    Gradient (and cost) of a trajectory's repulsion from a set of "sources".
    Each source = {'pos': (n+2, D), 'rad': float}:
      - static obstacle : 'pos' = the center repeated n+2 times
      - other agent     : 'pos' = its full trajectory (time-aligned)

    For each waypoint i:
        d  = ||x_i(t) - source(t)|| - rad        (signed distance to source)
        grad = ||v|| * (I - v_hat v_hat^T) * sum c'(d)*(x-src)/||x-src||
    (velocity weighting + projection perp. to velocity, as in Part I; the
     curvature term is omitted here -> more stable in the multi-agent case.)
    """
    n, D = traj.shape
    Q = np.vstack([start, traj, goal])
    grad = np.zeros((n, D))
    cost = 0.0
    I = np.eye(D)

    for i in range(n):
        x = Q[i + 1]
        v = (Q[i + 2] - Q[i]) / (2.0 * dt)
        speed = np.linalg.norm(v)

        gc = np.zeros(D)
        for src in sources:
            sx = src["pos"][i + 1]            # source position at the SAME t
            diff = x - sx
            dist = np.linalg.norm(diff) + 1e-9
            d = dist - src["rad"]
            cost += c_cost(d, eps) * speed * dt
            gc += c_grad(d, eps) * diff / dist

        if speed < 1e-6:
            grad[i] = gc
            continue
        vhat = v / speed
        P = I - np.outer(vhat, vhat)
        grad[i] = speed * (P @ gc)

    return grad, cost


# ============================================================================
# 2. ONE COVARIANT STEP for an agent (reuses Part I metric)
# ============================================================================
def step_agent(traj, start, goal, A, Ainv, b, dt,
               static, agents, lam, mu, eps_o, eps_a, step):
    """One covariant descent step for the agent, with:
       static = fixed obstacles, agents = other agents (moving sources)."""
    g_smooth = A @ traj + b
    g_obs, c_obs = repulsion_grad(traj, start, goal, dt, static, eps_o)
    g_agt, c_agt = repulsion_grad(traj, start, goal, dt, agents, eps_a)
    g = lam * g_smooth + g_obs + mu * g_agt
    traj = traj - step * (Ainv @ g)
    return traj, c_obs, c_agt


# ============================================================================
# 3. MULTI-AGENT SOLVER (the 3 strategies)
# ============================================================================
def solve_swarm(starts, goals, obstacles=(), strategy="gauss_seidel",
                n=50, dt=1.0, lam=0.6, mu=1.0, eps_o=1.5, eps_a=1.0,
                r_safe=1.0, step=0.02, iters=400, seed=0, omega=1.0):
    # omega = UNDER-RELAXATION (damping) factor for Jacobi.
    #   xi <- xi + omega*(covariant_step - xi). omega<1 calms the oscillations
    #   caused by simultaneous updates. omega=1 = undamped step.
    K = len(starts)
    A, Kf, Kb = build_smoothness(n, dt)
    Ainv = np.linalg.inv(A + 1e-9 * np.eye(n))
    KfKb = Kf.T @ Kb

    bs = [KfKb @ np.vstack([starts[k], goals[k]]) for k in range(K)]
    static = [{"pos": np.tile(o["c"], (n + 2, 1)), "rad": o["r"]} for o in obstacles]

    # init: straight lines + small noise (BREAKS symmetry, else deadlock)
    rng = np.random.default_rng(seed)
    trajs = []
    for k in range(K):
        line = np.linspace(starts[k], goals[k], n + 2)[1:-1].copy()
        line += rng.normal(0, 0.05, line.shape)
        trajs.append(line)

    def full(k, T=None):
        T = trajs[k] if T is None else T
        return np.vstack([starts[k], T, goals[k]])

    history = []

    if strategy == "priority":
        # fixed order: agent 0 first (free), then each avoids the FINAL ones
        finalized = []
        for k in range(K):
            t = trajs[k]
            agents = [{"pos": fq, "rad": r_safe} for fq in finalized]
            for _ in range(iters):
                t, _, _ = step_agent(t, starts[k], goals[k], A, Ainv, bs[k],
                                     dt, static, agents, lam, mu, eps_o, eps_a, step)
            trajs[k] = t
            finalized.append(full(k))
        n_iters_equiv = K * iters

    else:
        for it in range(iters):
            if strategy == "jacobi":
                snap = [full(k) for k in range(K)]          # snapshot of all
                new = []
                for k in range(K):
                    agents = [{"pos": snap[j], "rad": r_safe}
                              for j in range(K) if j != k]
                    t, _, _ = step_agent(trajs[k], starts[k], goals[k], A, Ainv,
                                         bs[k], dt, static, agents, lam, mu,
                                         eps_o, eps_a, step)
                    # under-relaxation: apply only a fraction omega of the step
                    new.append(trajs[k] + omega * (t - trajs[k]))
                trajs = new
            elif strategy == "gauss_seidel":
                for k in range(K):
                    agents = [{"pos": full(j), "rad": r_safe}
                              for j in range(K) if j != k]
                    trajs[k], _, _ = step_agent(trajs[k], starts[k], goals[k], A,
                                                Ainv, bs[k], dt, static, agents,
                                                lam, mu, eps_o, eps_a, step)
            history.append(system_cost(trajs, starts, goals, A, dt, static,
                                       lam, mu, eps_o, eps_a, r_safe))
        n_iters_equiv = iters

    return trajs, np.array(history), n_iters_equiv


# ============================================================================
# 4. METRICS
# ============================================================================
def system_cost(trajs, starts, goals, A, dt, static, lam, mu, eps_o, eps_a, r_safe):
    """Total system cost (sum over agents). Used to plot convergence."""
    K = len(trajs)
    total = 0.0
    for k in range(K):
        total += lam * 0.5 * np.sum((A @ trajs[k]) * trajs[k])
        _, c_o = repulsion_grad(trajs[k], starts[k], goals[k], dt, static, eps_o)
        agents = [{"pos": np.vstack([starts[j], trajs[j], goals[j]]), "rad": r_safe}
                  for j in range(K) if j != k]
        _, c_a = repulsion_grad(trajs[k], starts[k], goals[k], dt, agents, eps_a)
        total += c_o + mu * c_a
    return total


def min_inter_agent_dist(trajs, starts, goals):
    """Minimum distance between agent centers (over all time, all pairs)."""
    K = len(trajs)
    Qs = [np.vstack([starts[k], trajs[k], goals[k]]) for k in range(K)]
    dmin = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = np.min(np.linalg.norm(Qs[i] - Qs[j], axis=1))
            dmin = min(dmin, d)
    return dmin


def avg_path_length(trajs, starts, goals):
    K = len(trajs)
    tot = 0.0
    for k in range(K):
        Q = np.vstack([starts[k], trajs[k], goals[k]])
        tot += np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1))
    return tot / K


# ============================================================================
# 5. DEMO: antipodal swap (classic multi-agent benchmark)
# ============================================================================
if __name__ == "__main__":
    # K agents on a circle; each goal = diametrically opposite point.
    # All straight lines pass through the center AT THE SAME TIME -> max conflict.
    K = 6
    R = 5.0
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    starts = [np.array([R * np.cos(a), R * np.sin(a)]) for a in ang]
    goals = [-s for s in starts]            # opposite
    obstacles = []                           # pure inter-agent test

    r_safe = 1.0          # desired separation between centers
    r_col = 0.5           # collision if centers closer than this (agents r=0.25)

    results = {}
    strat_omega = {"jacobi": 0.4, "gauss_seidel": 1.0, "priority": 1.0}
    for strat in ["jacobi", "gauss_seidel", "priority"]:
        trajs, hist, nit = solve_swarm(
            starts, goals, obstacles, strategy=strat,
            n=50, dt=1.0, lam=0.6, mu=1.5, eps_a=1.2,
            r_safe=r_safe, step=0.02, iters=400, seed=1,
            omega=strat_omega[strat])
        dmin = min_inter_agent_dist(trajs, starts, goals)
        plen = avg_path_length(trajs, starts, goals)
        conv = None
        if len(hist):
            target = 1.02 * hist[-1]
            below = np.where(hist <= target)[0]
            conv = int(below[0]) if len(below) else len(hist)
        results[strat] = dict(trajs=trajs, hist=hist, nit=nit, dmin=dmin,
                              plen=plen, conv=conv)
        print(f"\n=== {strat.upper()} ===")
        print(f"  min inter-agent dist : {dmin:.3f}   "
              f"({'COLLISION' if dmin < r_col else 'collision-free'}; "
              f"{'margin OK' if dmin >= r_safe else 'below r_safe'})")
        print(f"  avg path length      : {plen:.3f}   (straight line = {2*R:.1f})")
        if len(hist):
            print(f"  final system cost U  : {hist[-1]:.3f}")
            print(f"  iters -> converged   : {conv}")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, K))
    for ax, strat in zip(axes, ["jacobi", "gauss_seidel", "priority"]):
        r = results[strat]
        for k in range(K):
            Q = np.vstack([starts[k], r["trajs"][k], goals[k]])
            ax.plot(Q[:, 0], Q[:, 1], color=colors[k], lw=2)
            ax.scatter(*starts[k], color=colors[k], s=35, zorder=5)
            ax.scatter(*goals[k], color=colors[k], s=35, marker="x", zorder=5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{strat}\nmin clearance = {r['dmin']:.2f} | "
                     f"avg length = {r['plen']:.2f}")
    fig.suptitle("COSMOS - Part II: CHOMP multi-agent (antipodal swap, K=6)\n"
                 "circle = start, cross = goal", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("chomp_part2_swarm.png", dpi=120)
    print("\nFigure -> chomp_part2_swarm.png")

    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results["jacobi"]["hist"], label="Jacobi", color="#d98", lw=2)
    ax.plot(results["gauss_seidel"]["hist"], label="Gauss-Seidel", color="#27a", lw=2)
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("iteration (full system sweep)")
    ax.set_ylabel("system cost U (log)")
    ax.set_title("Convergence: Jacobi vs Gauss-Seidel")
    fig2.tight_layout()
    fig2.savefig("chomp_part2_convergence.png", dpi=120)
    print("Figure -> chomp_part2_convergence.png")
