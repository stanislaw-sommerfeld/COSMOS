"""
COSMOS - unified entry point
============================================================================
ONE program, called once, that orchestrates the whole pipeline through flags.
It does NOT reimplement the science: it imports the already-validated building
blocks and composes them. The standalone scripts (chomp_*, cosmos_*) stay as
the pedagogical / figure-generating library; this file is the mission driver.

Reused (bit-exact) from the validated modules:
    chomp_cw   : cw_operators, cw_offset, to_flat, to_traj, cw_control, delta_v,
                 obstacle_gradient
    chomp_swarm: repulsion_grad        (soft inter-agent + obstacle)
    cosmos_safe: gramian, lq_opt       (LQ optimum / assignment cost)

What this adds (thin orchestration layer):
    - a unified multi-agent Gauss-Seidel solver with selectable safety projection
    - Hungarian assignment whose cost matrix is the closed-form LQ optimum
      (this is how Module A enters the mission)
    - scenarios: reconfig (formation -> formation), debris, single, swarm
    - SI unit reporting, 2D/3D, and a sim/results/both output mode

Examples
--------
    python cosmos.py                              # reconfig, fuel, riemannian, both
    python cosmos.py --metric free --safety none  # baseline ablation
    python cosmos.py --assign hungarian --units si
    python cosmos.py --scenario debris --dim 3 --mode results
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

# ---- validated library (single source of truth) ---------------------------
from .chomp_cw import (cw_operators, cw_offset, to_flat, to_traj,
                      cw_control, delta_v, obstacle_gradient)
from .chomp_swarm import repulsion_grad


def gramian(A, B, T, nq=4000):
    """Controllability Gramian W = int_0^T Phi(t) B B^T Phi(t)^T dt
       (same as cosmos_safe.py; inlined to avoid importing that script's body)."""
    dq = T / nq
    Phi = np.eye(A.shape[0])
    step = expm(A * dq)
    BBt = B @ B.T
    W = np.zeros_like(A, dtype=float)
    for _ in range(nq):
        W += Phi @ BBt @ Phi.T * dq
        Phi = step @ Phi
    return W


# ============================================================================
# 1. SI scaling (consistent with cosmos_units.py)
# ============================================================================
def si_constants():
    MU, RE, ALT = 3.986004418e14, 6.371e6, 500e3
    a = RE + ALT
    n = np.sqrt(MU / a ** 3)          # mean motion [rad/s]
    return dict(n=n, period_min=2 * np.pi / n / 60.0,
                L=100.0, K_DV=100.0 * n, alt_km=ALT / 1e3)


# ============================================================================
# 2. dim-general safety projection (generalizes cosmos_safe's 2D version)
# ============================================================================
def _normal(x, c):
    v = x - c
    d = np.linalg.norm(v) + 1e-9
    return v / d, d


def project_euclid_nd(traj, obstacles, margin=0.05):
    for i in range(len(traj)):
        for ob in obstacles:
            ni, d = _normal(traj[i], ob["c"])
            if d - ob["r"] < margin:
                traj[i] = ob["c"] + ni * (ob["r"] + margin)
    return traj


def project_metric_nd(traj, obstacles, Minv, margin=0.05, sweeps=3):
    """Riemannian projection in M = B^T B (dim-general). For each violated
       waypoint i: dxi = Minv a (margin - clr) / (a^T Minv a)."""
    n, D = traj.shape
    xi = to_flat(traj)
    for _ in range(sweeps):
        Tj = to_traj(xi, n)
        for i in range(n):
            for ob in obstacles:
                ni, d = _normal(Tj[i], ob["c"])
                clr = d - ob["r"]
                if clr < margin:
                    a = np.zeros(D * n)
                    for dd in range(D):
                        a[dd * n + i] = ni[dd]
                    Ma = Minv @ a
                    xi = xi + Ma * ((margin - clr) / (a @ Ma))
                    Tj = to_traj(xi, n)
    return to_traj(xi, n)


def apply_safety(traj, obstacles, Minv, mode):
    if not obstacles or mode == "none":
        return traj
    if mode == "euclid":
        return project_euclid_nd(traj, obstacles)
    return project_metric_nd(traj, obstacles, Minv)


# ============================================================================
# 3. Hungarian assignment with the closed-form LQ cost (Module A in the mission)
# ============================================================================
def cw_state_matrices(nm, dim):
    """Continuous CW state-space (A, B) for the LQ cost. 3D adds the decoupled
       cross-track oscillator z'' + n^2 z = u_z."""
    if dim == 2:
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                      [3 * nm ** 2, 0, 0, 2 * nm], [0, 0, -2 * nm, 0]], float)
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
    else:
        A = np.zeros((6, 6)); B = np.zeros((6, 3))
        A[0, 3] = A[1, 4] = A[2, 5] = 1.0
        A[3, 0] = 3 * nm ** 2; A[3, 4] = 2 * nm; A[4, 3] = -2 * nm
        A[5, 2] = -nm ** 2
        B[3, 0] = B[4, 1] = B[5, 2] = 1.0
    return A, B


def lq_cost_matrix(starts, goals, nm, T, dim):
    """C[i,j] = analytical LQ optimal energy to take agent i (at rest) to slot j
       (at rest), via the controllability Gramian. Closed form, cheap."""
    A, B = cw_state_matrices(nm, dim)
    W = gramian(A, B, T)
    Winv = np.linalg.inv(W)
    PhiT = expm(A * T)
    m = len(starts)
    C = np.zeros((m, m))
    for i in range(m):
        s0 = np.concatenate([starts[i], np.zeros(dim)])
        for j in range(m):
            sf = np.concatenate([goals[j], np.zeros(dim)])
            ds = sf - PhiT @ s0
            C[i, j] = ds @ (Winv @ ds)
    return C


def assign(starts, goals, mode, nm, T, dim):
    """Return goals reordered so goals[i] is the slot assigned to agent i."""
    if mode == "fixed" or len(starts) <= 1:
        return list(goals), None
    C = lq_cost_matrix(starts, goals, nm, T, dim)
    _, col = linear_sum_assignment(C)
    return [goals[j] for j in col], C


# ============================================================================
# 4. unified multi-agent solver (Gauss-Seidel, covariant in the chosen metric)
# ============================================================================
def solve(starts, goals, nm, n, dt, dim,
          obstacles=(), lam=1.0, mu=2.0, eps_o=0.8, eps_a=1.0, r_safe=0.6,
          w_bc=20000.0, step=0.02, iters=900, safety="riemann"):
    """nm=0 -> free-space metric (A^T A); nm>0 -> fuel metric (B^T B).
       Always scored under the TRUE dynamics afterwards by the caller."""
    starts = [np.asarray(s, float) for s in starts]
    goals = [np.asarray(g, float) for g in goals]
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc, dim=dim)
    c0 = [cw_offset(blocks, starts[k], goals[k]) for k in range(len(starts))]

    trajs = [np.linspace(starts[k], goals[k], n + 2)[1:-1].copy()
             for k in range(len(starts))]

    for _ in range(iters):
        for k in range(len(trajs)):                       # Gauss-Seidel
            xi = to_flat(trajs[k])
            g = lam * (B.T @ (B @ xi + c0[k]))            # fuel / smoothness
            # static keep-out (soft) + inter-agent repulsion (soft)
            srcs = []
            for ob in obstacles:
                srcs.append({"pos": np.tile(ob["c"], (n + 2, 1)), "rad": ob["r"]})
            for j in range(len(trajs)):
                if j == k:
                    continue
                Qj = np.vstack([starts[j], trajs[j], goals[j]])
                srcs.append({"pos": Qj, "rad": r_safe})
            if srcs:
                g_rep, _ = repulsion_grad(trajs[k], starts[k], goals[k],
                                          dt, srcs, eps_a)
                g = g + mu * to_flat(g_rep)
            trajs[k] = to_traj(xi - step * (Minv @ g), n)
            trajs[k] = apply_safety(trajs[k], obstacles, Minv, safety)
    return trajs


# ============================================================================
# 5. metrics
# ============================================================================
def total_dv(trajs, starts, goals, dt, nm_true):
    dv = e = 0.0
    per = []
    for k in range(len(trajs)):
        d, en = delta_v(trajs[k], starts[k], goals[k], dt, nm_true)
        dv += d; e += en; per.append(d)
    return dv, e, per


def min_pair_clearance(trajs, starts, goals):
    Q = [np.vstack([starts[k], trajs[k], goals[k]]) for k in range(len(trajs))]
    m = len(Q)
    best = np.inf
    for i in range(m):
        for j in range(i + 1, m):
            best = min(best, np.min(np.linalg.norm(Q[i] - Q[j], axis=1)))
    return best


# ============================================================================
# 6. scenarios
# ============================================================================
def scenario(name, dim):
    """Return (starts, goals, obstacles, title)."""
    if name == "reconfig":
        # launch 'train' (aligned in along-track y, radial x=0) -> passive 2:1
        # relative-orbit slots (Hill ellipse): x = rho cos, y = -2 rho sin.
        rho = 3.0
        ph = np.deg2rad([0, 90, 180, 270])
        starts = [np.array([0.0, v]) for v in (-3.0, -1.0, 1.0, 3.0)]
        goals = [np.array([rho * np.cos(p), -2 * rho * np.sin(p)]) for p in ph]
        obs = [{"c": np.array([0.0, 0.0]), "r": 1.0}]   # central keep-out
        title = "Formation reconfiguration: launch train -> passive 2:1 orbit"
    elif name == "debris":
        starts = [np.array([x, -6.0]) for x in (-1.5, -0.5, 0.5, 1.5)]
        goals = [np.array(g) for g in
                 [(-4.0, 3.0), (4.0, 3.0), (-3.0, -2.0), (3.0, -2.0)]]
        obs = []
        title = "Multi-chaser debris rendezvous"
    elif name == "swarm":
        starts = [np.array([-5.0, v]) for v in (-2.0, 0.0, 2.0)]
        goals = [np.array([5.0, -v]) for v in (-2.0, 0.0, 2.0)]
        obs = [{"c": np.array([0.0, 0.0]), "r": 1.2}]
        title = "Swarm crossing"
    else:  # single
        starts = [np.array([3.0, 5.0])]
        goals = [np.array([0.0, 0.0])]
        obs = [{"c": np.array([1.0, 2.6]), "r": 1.3}]
        title = "Single-agent transfer"

    if dim == 3:                                          # lift into cross-track
        starts = [np.append(s, z) for s, z in
                  zip(starts, np.linspace(-1, 1, len(starts)))]
        goals = [np.append(g, 0.0) for g in goals]
        obs = [{"c": np.append(o["c"], 0.0), "r": o["r"]} for o in obs]
    return starts, goals, obs, title


# ============================================================================
# 7. figure
# ============================================================================
def plot(trajs_fuel, trajs_free, starts, goals, obstacles, dim, K_X,
         title, path):
    fig = plt.figure(figsize=(7.5, 6.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(starts)))
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        for k in range(len(trajs_fuel)):
            Q = np.vstack([starts[k], trajs_fuel[k], goals[k]]) * K_X
            ax.plot(Q[:, 1], Q[:, 0], Q[:, 2], color=colors[k], lw=2.2)
            ax.scatter(*[[starts[k][d] * K_X] for d in (1, 0, 2)],
                       color=colors[k], s=40)
            ax.scatter(*[[goals[k][d] * K_X] for d in (1, 0, 2)],
                       color=colors[k], marker="*", s=150)
        ax.set_xlabel("along-track y"); ax.set_ylabel("radial x")
        ax.set_zlabel("cross-track z")
    else:
        ax = fig.add_subplot(111)
        th = np.linspace(0, 2 * np.pi, 120)
        for ob in obstacles:
            ax.fill(ob["c"][1] * K_X + ob["r"] * K_X * np.cos(th),
                    ob["c"][0] * K_X + ob["r"] * K_X * np.sin(th),
                    color="#d96459", alpha=0.18)
        for k in range(len(trajs_fuel)):
            Qf = np.vstack([starts[k], trajs_free[k], goals[k]]) * K_X
            Qu = np.vstack([starts[k], trajs_fuel[k], goals[k]]) * K_X
            ax.plot(Qf[:, 1], Qf[:, 0], "--", color="#bbb", lw=1.3)
            ax.plot(Qu[:, 1], Qu[:, 0], "-", color=colors[k], lw=2.4)
            ax.scatter(starts[k][1] * K_X, starts[k][0] * K_X,
                       color=colors[k], s=45, zorder=5)
            ax.scatter(goals[k][1] * K_X, goals[k][0] * K_X,
                       color=colors[k], marker="*", s=160, zorder=5)
        ax.plot([], [], "--", color="#bbb", label="free-space")
        ax.plot([], [], "-", color="#333", label="fuel-aware")
        ax.set_xlabel("along-track y"); ax.set_ylabel("radial x")
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    return path


# ============================================================================
# 8. CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="COSMOS unified mission driver.")
    p.add_argument("--scenario", choices=["reconfig", "debris", "swarm", "single"],
                   default="reconfig")
    p.add_argument("--metric", choices=["free", "fuel"], default="fuel")
    p.add_argument("--safety", choices=["none", "euclid", "riemann"],
                   default="riemann")
    p.add_argument("--assign", choices=["fixed", "hungarian"], default="hungarian")
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--units", choices=["norm", "si"], default="norm")
    p.add_argument("--mode", choices=["sim", "results", "both"], default="both")
    p.add_argument("--iters", type=int, default=900)
    p.add_argument("--out", default="cosmos_run.png")
    args = p.parse_args()

    nm_true = 1.0                              # normalized true mean motion
    T = np.pi; N = 50; dt = T / (N + 1)
    si = si_constants()
    K_DV = si["K_DV"] if args.units == "si" else 1.0
    K_X = si["L"] if args.units == "si" else 1.0
    unit_dv = "m/s" if args.units == "si" else "(norm)"

    starts, goals0, obstacles, title = scenario(args.scenario, args.dim)
    goals, C = assign(starts, goals0, args.assign, nm_true, T, args.dim)

    nm_solve = nm_true if args.metric == "fuel" else 0.0
    common = dict(nm=nm_solve, n=N, dt=dt, dim=args.dim, obstacles=obstacles,
                  safety=args.safety, iters=args.iters)
    trajs = solve(starts, goals, **common)
    # always also compute the free-space baseline for the comparison figure/number
    trajs_free = trajs if args.metric == "free" else \
        solve(starts, goals, **{**common, "nm": 0.0})

    dv, en, per = total_dv(trajs, starts, goals, dt, nm_true)
    dvf, enf, _ = total_dv(trajs_free, starts, goals, dt, nm_true)
    clr = min_pair_clearance(trajs, starts, goals)

    if args.mode in ("results", "both"):
        print(f"=== COSMOS | scenario={args.scenario} metric={args.metric} "
              f"safety={args.safety} assign={args.assign} dim={args.dim}D "
              f"units={args.units} ===")
        print(f"{title}")
        if args.units == "si":
            print(f"LEO {si['alt_km']:.0f} km | period {si['period_min']:.1f} min "
                  f"| scene scale {si['L']:.0f} m")
        if C is not None:
            print(f"assignment            : Hungarian on LQ cost "
                  f"(total LQ cost {C[np.arange(len(starts)), :].min():.2f}-scale)")
        print(f"fuel-aware delta-v    : {dv * K_DV:8.3f} {unit_dv}  "
              f"(energy {en:.2f})")
        print(f"free-space delta-v    : {dvf * K_DV:8.3f} {unit_dv}  "
              f"(energy {enf:.2f})")
        if dvf > 1e-9:
            print(f"saving                : {100 * (dvf - dv) / dvf:+5.1f} %")
        print(f"per-agent delta-v     : "
              f"{', '.join(f'{x * K_DV:.3f}' for x in per)} {unit_dv}")
        print(f"min inter-agent clear.: {clr * K_X:+.3f} "
              f"{'m' if args.units == 'si' else '(norm)'}  "
              f"({'COLLISION-FREE' if clr > 0 else 'COLLISION'})")

    if args.mode in ("sim", "both"):
        out = plot(trajs, trajs_free, starts, goals, obstacles,
                   args.dim, K_X, title, args.out)
        print(f"figure -> {out}")


if __name__ == "__main__":
    main()
