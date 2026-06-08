"""
COSMOS - CHOMP Part III: fuel-aware metric in the Hill frame (Clohessy-Wiltshire)
================================================================================

THE contribution. In free space, smoothness penalizes the raw acceleration
(metric A^T A). In orbit the real thrust is NOT the raw acceleration:

    Clohessy-Wiltshire (2D Hill frame, x = radial, y = along-track):
        x_ddot - 2 n y_dot - 3 n^2 x = u_x
        y_ddot + 2 n x_dot           = u_y
    =>  u = xi_ddot - f_CW(xi, xi_dot)        (the actual control / thrust)

So standard CHOMP minimizes the WRONG quantity. We fix it: penalize u.
Because f_CW is linear in (position, velocity), u = B xi + c0 with

    B = [[ D2 - 3 n^2 I ,   -2 n D1 ],
         [  2 n D1       ,    D2     ]]      (D2 = accel op, D1 = velocity op)

f_CW couples x and y -> B is NOT block-diagonal. The metric becomes B^T B and
the preconditioner (B^T B)^-1. Everything else (covariant descent, obstacles)
is unchanged. Free space = this code with n_orbit = 0 (B -> blockdiag(D2, D2)).

We measure the actual delta-v (L1) and energy (L2) under the TRUE dynamics, for
a free-space plan vs a fuel-aware plan, and expect the fuel-aware one to be
cheaper because it exploits the natural orbital drift.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_single_agent import obstacle_gradient   # reused workspace gradient


# ============================================================================
# 1. CW operators -> B (control operator) and the metric B^T B
# ============================================================================
def cw_operators(n, dt, nm, w_bc=0.0, dim=2):
    """Build B (control operator), (B^T B)^-1, and boundary blocks.
       nm  = mean motion (nm=0 recovers free-space / raw acceleration).
       dim = 2 (radial, along-track) or 3 (+ cross-track z).
       w_bc = weight of the rendezvous boundary condition (zero endpoint
              velocity). w_bc=0 -> no BC. When w_bc>0, 2*dim rows are appended
              to B so the metric itself penalizes the start/arrival velocities.
       CW dynamics:  x'' - 2n y' - 3n^2 x = u_x ; y'' + 2n x' = u_y ;
                     z'' + n^2 z = u_z  (cross-track is a decoupled oscillator)."""
    # full operators over augmented sequence [start, q_1..q_n, goal] (len n+2)
    D2 = np.zeros((n, n + 2))   # second difference (acceleration)
    D1 = np.zeros((n, n + 2))   # centered first difference (velocity)
    P0 = np.zeros((n, n + 2))   # interior position selector
    for t in range(n):
        D2[t, t] = 1.0; D2[t, t + 1] = -2.0; D2[t, t + 2] = 1.0
        D1[t, t] = -1.0; D1[t, t + 2] = 1.0
        P0[t, t + 1] = 1.0
    D2 /= dt ** 2
    D1 /= (2.0 * dt)

    def split(Mfull):
        return Mfull[:, 1:n + 1], Mfull[:, [0, n + 1]]   # free (n x n), bound (n x 2)

    D2f, D2b = split(D2)
    D1f, D1b = split(D1)
    P0f, P0b = split(P0)        # P0f = I_n, P0b = 0

    I = np.eye(n)
    Z = np.zeros((n, n))
    D2f3 = D2f + nm ** 2 * I            # cross-track block: z'' + n^2 z
    Bxx = D2f - 3.0 * nm ** 2 * I
    Bxy = -2.0 * nm * D1f
    Byx = 2.0 * nm * D1f
    Byy = D2f
    if dim == 2:
        B = np.block([[Bxx, Bxy], [Byx, Byy]])
    elif dim == 3:
        B = np.block([[Bxx, Bxy, Z],
                      [Byx, Byy, Z],
                      [Z,   Z,   D2f3]])
    else:
        raise ValueError("dim must be 2 or 3")

    sw = np.sqrt(w_bc)
    if w_bc > 0.0:
        # 2*dim boundary-velocity rows (start/goal per dimension):
        #   v_start_d = (q_1,d - start_d)/dt ,  v_goal_d = (goal_d - q_n,d)/dt
        R = np.zeros((2 * dim, dim * n))
        for d in range(dim):
            R[2 * d,     d * n]         = sw / dt        # start velocity, dim d
            R[2 * d + 1, d * n + n - 1] = -sw / dt       # goal velocity,  dim d
        B = np.vstack([B, R])

    M = B.T @ B + 1e-9 * np.eye(dim * n)
    Minv = np.linalg.inv(M)

    blocks = dict(D2b=D2b, D1b=D1b, P0b=P0b, nm=nm, dt=dt, n=n,
                  sw=sw, w_bc=w_bc, dim=dim)
    return B, Minv, blocks


def cw_offset(blocks, start, goal):
    """Constant term c0 from the fixed endpoints, so that u = B xi + c0.
       Generalized to dim=2 or 3. Appends 2*dim BC constants when w_bc>0."""
    D2b, D1b, nm, dim = blocks["D2b"], blocks["D1b"], blocks["nm"], blocks["dim"]
    e = [np.array([start[d], goal[d]]) for d in range(dim)]   # per-dim [start,goal]
    c0x = D2b @ e[0] - 2.0 * nm * (D1b @ e[1])                # radial
    c0y = D2b @ e[1] + 2.0 * nm * (D1b @ e[0])                # along-track
    parts = [c0x, c0y]
    if dim == 3:
        parts.append(D2b @ e[2] + nm ** 2 * (np.zeros(2) @ e[2]))  # cross-track
    c0 = np.concatenate(parts)
    if blocks["w_bc"] > 0.0:
        sw, dt = blocks["sw"], blocks["dt"]
        bc = []
        for d in range(dim):
            bc += [-sw * start[d] / dt, sw * goal[d] / dt]
        c0 = np.concatenate([c0, np.array(bc)])
    return c0


# ============================================================================
# 2. helpers: flatten between (n,D) and block layout [x.., y.., (z..)]
# ============================================================================
def to_flat(traj):
    D = traj.shape[1]
    return np.concatenate([traj[:, d] for d in range(D)])

def to_traj(flat, n):
    D = len(flat) // n
    return np.column_stack([flat[d * n:(d + 1) * n] for d in range(D)])


# ============================================================================
# 3. CW-CHOMP optimizer (single agent), covariant w.r.t. the fuel metric
# ============================================================================
def chomp_cw(start, goal, nm, n=60, dt=1.0, lam=1.0,
             obstacle=None, eps=1.0, step=0.02, iters=400, w_bc=0.0):
    start = np.asarray(start, float); goal = np.asarray(goal, float)
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc)
    c0 = cw_offset(blocks, start, goal)

    traj = np.linspace(start, goal, n + 2)[1:-1].copy()
    init = traj.copy()
    hist = []
    for _ in range(iters):
        xi = to_flat(traj)
        U = B @ xi + c0
        g_fuel = B.T @ U                                  # grad of 1/2||u||^2
        g = lam * g_fuel
        if obstacle is not None:
            g_obs, _ = obstacle_gradient(traj, start, goal, dt,
                                         obstacle["c"], obstacle["r"], eps)
            g = g + to_flat(g_obs)
        xi = xi - step * (Minv @ g)
        traj = to_traj(xi, n)
        hist.append(0.5 * np.sum(U ** 2))
    return traj, init, np.array(hist)


# ============================================================================
# 3b. GENERALITY: B = D2 - G for ANY linear relative dynamics (plug in G)
# ============================================================================
def cw_coeffs(nm):
    """Clohessy-Wiltshire in-plane coefficients (radial stiffness, Coriolis rate)."""
    return (3.0 * nm ** 2, 2.0 * nm)


def ss_coeffs(nm, inc_rad, alt_m=500e3, RE=6.378137e6, J2=1.08263e-3):
    """Schweighart-Sedwick (J2) in-plane coefficients. Same shape as CW, with the
       J2 correction c = sqrt(1+s), s = (3 J2 Re^2 / 8 a^2)(1 + 3 cos 2i). As J2->0,
       c->1 and these reduce to cw_coeffs(nm) exactly."""
    a = RE + alt_m
    s = (3.0 * J2 * RE ** 2) / (8.0 * a ** 2) * (1.0 + 3.0 * np.cos(2.0 * inc_rad))
    c = np.sqrt(1.0 + s)
    return ((5.0 * c ** 2 - 2.0) * nm ** 2, 2.0 * nm * c)


def linear_relative_operators(n, dt, kxx, kc, w_bc=0.0):
    """Control operator B = D2 - G for ANY linear, constant-coefficient, in-plane
       relative dynamics of the form
           x'' - kc*y' - kxx*x = u_x ,    y'' + kc*x' = u_y       (2D Hill frame)
       This is the whole 'metric = dynamics' claim, in code: the smoothness operator
       is built from the dynamics you hand it. cw_operators(nm, dim=2) is exactly the
       special case (kxx, kc) = cw_coeffs(nm); J2 is (kxx, kc) = ss_coeffs(nm, i).
       Returns (B, (B^T B)^-1, blocks) just like cw_operators."""
    D2 = np.zeros((n, n + 2)); D1 = np.zeros((n, n + 2))
    for t in range(n):
        D2[t, t] = 1.0; D2[t, t + 1] = -2.0; D2[t, t + 2] = 1.0
        D1[t, t] = -1.0; D1[t, t + 2] = 1.0
    D2 /= dt ** 2; D1 /= (2.0 * dt)
    D2f, D2b = D2[:, 1:n + 1], D2[:, [0, n + 1]]
    D1f, D1b = D1[:, 1:n + 1], D1[:, [0, n + 1]]
    I = np.eye(n)
    Bxx = D2f - kxx * I
    Bxy = -kc * D1f
    Byx = kc * D1f
    Byy = D2f
    B = np.block([[Bxx, Bxy], [Byx, Byy]])
    sw = np.sqrt(w_bc)
    if w_bc > 0.0:
        R = np.zeros((4, 2 * n))
        R[0, 0] = sw / dt; R[1, n - 1] = -sw / dt           # radial start/goal vel
        R[2, n] = sw / dt; R[3, 2 * n - 1] = -sw / dt        # along-track start/goal vel
        B = np.vstack([B, R])
    M = B.T @ B + 1e-9 * np.eye(2 * n)
    Minv = np.linalg.inv(M)
    blocks = dict(D2b=D2b, D1b=D1b, kc=kc, dt=dt, n=n, sw=sw, w_bc=w_bc)
    return B, Minv, blocks


def linear_relative_offset(blocks, start, goal):
    """Constant term c0 for the generalized operator (depends on the coupling kc only;
       the stiffness acts on interior points and has no boundary contribution)."""
    D2b, D1b, kc = blocks["D2b"], blocks["D1b"], blocks["kc"]
    ex = np.array([start[0], goal[0]]); ey = np.array([start[1], goal[1]])
    c0x = D2b @ ex - kc * (D1b @ ey)
    c0y = D2b @ ey + kc * (D1b @ ex)
    c0 = np.concatenate([c0x, c0y])
    if blocks["w_bc"] > 0.0:
        sw, dt = blocks["sw"], blocks["dt"]
        c0 = np.concatenate([c0, np.array([-sw * start[0] / dt, sw * goal[0] / dt,
                                           -sw * start[1] / dt, sw * goal[1] / dt])])
    return c0


# ============================================================================
# 3c. closed-form unconstrained optimum (validation / warm-start, NOT a replacement)
# ============================================================================
def cw_unconstrained_optimum(start, goal, nm, n=60, dt=1.0, w_bc=0.0):
    """Exact minimum of the NO-OBSTACLE fuel cost 1/2 ||B xi + c0||^2.

    The fuel term is quadratic, so its minimizer is the single linear solve
        (B^T B) xi = -B^T c0      ->      xi = -(B^T B)^-1 B^T c0
    i.e. exactly the point covariant gradient descent converges to when there is
    no obstacle. Use it as (a) a ground-truth check that the descent has actually
    converged, or (b) a warm start. It does NOT replace CHOMP: with obstacles or
    multiple agents the cost is no longer quadratic and there is no closed form --
    that non-convex case is precisely what the covariant gradient descent is for.
    """
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc)
    c0 = cw_offset(blocks, start, goal)
    xi = -Minv @ (B.T @ c0)
    return to_traj(xi, n)


# ============================================================================
# 4. true delta-v / energy under CW dynamics (evaluation metric)
# ============================================================================
def cw_control(traj, start, goal, dt, nm):
    """Per-waypoint thrust u_t = a_t - f_CW(x_t, v_t) under true dynamics.
       f_CW = [2n vy + 3n^2 x, -2n vx, -n^2 z] (last term only if 3D)."""
    Q = np.vstack([start, traj, goal])
    n, D = traj.shape
    U = np.zeros((n, D))
    for t in range(n):
        x = Q[t + 1]
        v = (Q[t + 2] - Q[t]) / (2.0 * dt)
        a = (Q[t + 2] - 2.0 * Q[t + 1] + Q[t]) / dt ** 2
        f = np.zeros(D)
        f[0] = 2 * nm * v[1] + 3 * nm ** 2 * x[0]
        f[1] = -2 * nm * v[0]
        if D == 3:
            f[2] = -nm ** 2 * x[2]
        U[t] = a - f
    return U


def delta_v(traj, start, goal, dt, nm):
    U = cw_control(traj, start, goal, dt, nm)
    norms = np.linalg.norm(U, axis=1)
    dv = np.sum(norms) * dt                 # L1 (impulse proxy ~ fuel)
    energy = np.sum(norms ** 2) * dt        # L2 (energy)
    return dv, energy


# ============================================================================
# 5. EXPERIMENT: free-space plan vs fuel-aware plan, scored on true delta-v
# ============================================================================
if __name__ == "__main__":
    nm_true = 1.0                     # true mean motion (normalized units)
    T = np.pi                         # maneuver duration ~ half an orbit (strong drift)
    N = 60
    dt = T / (N + 1)
    start = np.array([3.0, 5.0])      # relative position in the Hill frame
    goal = np.array([0.0, 0.0])       # rendezvous to the target at the origin

    def run_scenario(obstacle, eps=0.8):
        tf, _, _ = chomp_cw(start, goal, nm=0.0, n=N, dt=dt,
                            obstacle=obstacle, eps=eps, iters=400)
        tu, _, h = chomp_cw(start, goal, nm=nm_true, n=N, dt=dt,
                            obstacle=obstacle, eps=eps, iters=400)
        dvf, ef = delta_v(tf, start, goal, dt, nm_true)
        dvu, eu = delta_v(tu, start, goal, dt, nm_true)
        return tf, tu, h, (dvf, ef), (dvu, eu)

    # Scenario A: free transfer (no obstacle) -> clean illustration
    tfA, tuA, hA, (dvfA, _), (dvuA, _) = run_scenario(None)
    # Scenario B: a keep-out zone blocks the natural coast -> realistic delta-v
    obs = {"c": np.array([1.0, 2.6]), "r": 1.3}
    tfB, tuB, hB, (dvfB, _), (dvuB, _) = run_scenario(obs, eps=0.8)

    print("=== Part III: fuel-aware vs free-space (scored under true CW) ===")
    print(f"Scenario A (no keep-out):")
    print(f"  free-space delta-v = {dvfA:.3f}   fuel-aware delta-v = {dvuA:.3f}"
          f"   -> {100*(dvfA-dvuA)/dvfA:+.0f}%  (fuel-aware ~ natural coast)")
    print(f"Scenario B (keep-out blocks the coast):")
    print(f"  free-space delta-v = {dvfB:.3f}   fuel-aware delta-v = {dvuB:.3f}"
          f"   -> {100*(dvfB-dvuB)/dvfB:+.0f}%  (both pay, fuel-aware cheaper)")
    # sanity: nm=0 scored under nm=0 must be ~0 for the straight line
    dv0, _ = delta_v(tfA, start, goal, dt, 0.0)
    print(f"  [sanity] free-space plan scored at nm=0: delta-v = {dv0:.4f} (~0 expected)")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, (tf, tu, dvf, dvu, ob, ttl) in zip(
            axes,
            [(tfA, tuA, dvfA, dvuA, None, "A: free transfer"),
             (tfB, tuB, dvfB, dvuB, obs, "B: with keep-out")]):
        Qf = np.vstack([start, tf, goal]); Qu = np.vstack([start, tu, goal])
        if ob is not None:
            th = np.linspace(0, 2*np.pi, 120)
            ax.fill(ob["c"][1] + ob["r"]*np.cos(th), ob["c"][0] + ob["r"]*np.sin(th),
                    color="#d96459", alpha=0.30)
        ax.plot(Qf[:, 1], Qf[:, 0], "--", color="#888", lw=2,
                label=f"free-space  (dv={dvf:.2f})")
        ax.plot(Qu[:, 1], Qu[:, 0], "-", color="#1d9e75", lw=2.6,
                label=f"fuel-aware  (dv={dvu:.2f})")
        ax.scatter(start[1], start[0], c="k", zorder=5, label="start")
        ax.scatter(goal[1], goal[0], c="r", marker="*", s=130, zorder=5, label="target")
        ax.set_xlabel("along-track  y"); ax.set_ylabel("radial  x")
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_title(ttl)
    fig.suptitle("COSMOS - Part III: fuel-aware CHOMP in the Hill frame "
                 "(Clohessy-Wiltshire)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("chomp_part3_cw.png", dpi=120)
    print("\nFigure -> chomp_part3_cw.png")
