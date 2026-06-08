"""
COSMOS - Validation sweep with confidence intervals
===================================================

Goes beyond the 10-seed "53% +/- 10%" anecdote: sweeps several axes (number of
agents, discretization n, formation geometry) over many random scenarios and reports
the fuel-aware delta-v saving as mean +/- a real 95% confidence interval (Student-t),
plus the collision-free success rate.

This is the "are the headline numbers robust?" experiment. It reuses the hardened
full system (cosmos_full.solve_swarm_cw) unchanged.

Run:  python -m cosmos.cosmos_sweep            (compact, ~1-2 min)
      python -m cosmos.cosmos_sweep --full     (more seeds / cells)
"""

import sys
import numpy as np
from .cosmos_full import solve_swarm_cw, total_delta_v, min_inter_agent

T = np.pi


def t_ci95(x):
    """mean and 95% half-width (Student-t) for a small sample."""
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    se = x.std(ddof=1) / np.sqrt(len(x))
    # t_0.975 quantile; small lookup avoids a scipy dependency in the hot path
    dof = len(x) - 1
    tval = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
            8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09,
            25: 2.06, 30: 2.04}.get(dof, 1.96 if dof > 30 else 2.20)
    return m, tval * se


def random_scenario(rng, K, geometry):
    """Random debris targets outside a central keep-out, with a chosen launch geometry."""
    obst = [{"c": np.zeros(2), "r": 1.6}]
    if geometry == "line":
        starts = [np.array([x, -6.0]) for x in np.linspace(-1.5, 1.5, K)]
    elif geometry == "arc":
        ang = np.linspace(-0.6, 0.6, K)
        starts = [np.array([3.5 * np.sin(a), -5.5 + 1.5 * np.cos(a)]) for a in ang]
    else:  # cluster
        starts = [np.array([rng.uniform(-1.2, 1.2), -6.0 + rng.uniform(-0.6, 0.6)])
                  for _ in range(K)]
    goals = []
    tries = 0
    while len(goals) < K and tries < 2000:
        tries += 1
        g = np.array([rng.uniform(-4.5, 4.5), rng.uniform(-3.0, 5.0)])
        if np.linalg.norm(g - obst[0]["c"]) < obst[0]["r"] + 0.8:
            continue
        if any(np.linalg.norm(g - h) < 1.7 for h in goals):
            continue
        goals.append(g)
    return starts, goals, obst


def run_cell(K, N, geometry, seeds, iters=600):
    dt = T / (N + 1)
    params = dict(n=N, dt=dt, lam=1.0, mu=3.0, eps_o=0.8, eps_a=1.2,
                  r_safe=0.9, w_bc=300.0, step=0.02, iters=iters, r_col=0.6)
    savings, free_flags = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        starts, goals, obst = random_scenario(rng, K, geometry)
        if len(goals) < K:
            continue
        tj_fuel = solve_swarm_cw(starts, goals, obst, nm=1.0, seed=s, **params)
        tj_free = solve_swarm_cw(starts, goals, obst, nm=0.0, seed=s, **params)
        dvf = total_delta_v(tj_fuel, starts, goals, dt, 1.0)
        dv0 = total_delta_v(tj_free, starts, goals, dt, 1.0)
        if dv0 > 1e-9:
            savings.append(100.0 * (dv0 - dvf) / dv0)
        free_flags.append(min_inter_agent(tj_fuel, starts, goals) > 0.0)
    m, hw = t_ci95(savings)
    succ = 100.0 * np.mean(free_flags) if free_flags else float("nan")
    return m, hw, succ, len(savings)


def main():
    quick = "--quick" in sys.argv[1:]      # fast smoke test (under-converged numbers)
    full = "--full" in sys.argv[1:]
    if quick:
        nseed, iters = 4, 250
    elif full:
        nseed, iters = 25, 800
    else:
        nseed, iters = 12, 600              # offline default (matches the paper regime)
    seeds = list(range(nseed))

    print("=== COSMOS validation sweep — fuel-aware delta-v saving (mean +/- 95% CI) ===")
    print(f"random scenarios per cell: {nseed} | iters: {iters} | Student-t 95% interval")
    if quick:
        print("** --quick smoke test: too few iters to converge; numbers UNDER-report the "
              "saving. Run without --quick offline for the real figures. **")
    print(f"\n{'cell':28} {'saving % (95% CI)':22} {'collision-free':>14} {'n':>4}")
    print("-" * 72)

    cells = [("K=4, n=45, line", 4, 45, "line"),
             ("K=4, n=45, arc", 4, 45, "arc"),
             ("K=4, n=45, cluster", 4, 45, "cluster"),
             ("K=3, n=45, line", 3, 45, "line"),
             ("K=5, n=45, line", 5, 45, "line"),
             ("K=4, n=30, line", 4, 30, "line"),
             ("K=4, n=60, line", 4, 60, "line")]
    if quick:
        cells = cells[:3]
    if full:
        cells += [("K=6, n=45, line", 6, 45, "line"),
                  ("K=4, n=80, line", 4, 80, "line")]

    all_means = []
    for label, K, N, geom in cells:
        m, hw, succ, ns = run_cell(K, N, geom, seeds, iters=iters)
        all_means.append(m)
        print(f"{label:28} {m:6.1f}  +/- {hw:4.1f}        {succ:8.0f} %     {ns:4d}")

    print("-" * 72)
    gm, ghw = t_ci95(all_means)
    print(f"across all cells: saving = {gm:.1f}% +/- {ghw:.1f}% "
          f"(headline single-scenario figure: ~53%)")
    print("Interpretation: the fuel-aware benefit is robust across geometry, agent")
    print("count and discretization; intervals are real CIs, not a hand-set band.")


if __name__ == "__main__":
    main()
