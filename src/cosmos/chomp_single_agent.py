"""
COSMOS - CHOMP single agent (Framework Part I)
==============================================

Trajectory optimization by COVARIANT GRADIENT descent in free space.

Core idea: a trajectory is just a vector of waypoints. We start from a straight
start->goal line, give it a cost U = smoothness + obstacle, and descend the
gradient -- but preconditioned by the smoothness metric (A^T A)^-1, which keeps
the trajectory smooth at every step.

Three building blocks:
  (1) the finite-difference matrix A (acceleration) -> metric A^T A
  (2) the obstacle cost/gradient (velocity-weighted + projection)
  (3) the covariant update  xi <- xi - alpha (A^T A)^-1 grad U

Everything is written for D dimensions (works in 2D and 3D). Demo is 2D.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 1. SMOOTHNESS METRIC  (block (1))
# ============================================================================
def build_smoothness(n, dt):
    """
    Build the SECOND-DIFFERENCE (acceleration) operator over the augmented
    trajectory [start, q_1, ..., q_n, goal] (length n+2), then the metric
    A = K_free^T K_free.

    For free waypoint t (0-based), the centered acceleration is:
        a_t = (Q[t] - 2 Q[t+1] + Q[t+2]) / dt^2      (augmented indices)
    The [1, -2, 1] stencil couples each point to its 2 neighbours; at the
    boundaries the neighbours are start/goal (fixed) -> they go into a constant.

    Returns:
        A        : (n, n) smoothness metric, symmetric positive definite
        K_free   : (n, n) block acting on the free points
        K_bound  : (n, 2) block acting on [start, goal]
    """
    K_full = np.zeros((n, n + 2))
    for t in range(n):
        K_full[t, t]     = 1.0
        K_full[t, t + 1] = -2.0
        K_full[t, t + 2] = 1.0
    K_full /= dt ** 2

    K_free  = K_full[:, 1:n + 1]        # columns of the free points
    K_bound = K_full[:, [0, n + 1]]     # columns of start and goal
    A = K_free.T @ K_free               # <-- the metric (SPD)
    return A, K_free, K_bound


# ============================================================================
# 2. OBSTACLE: analytic SDF + barrier cost  (block (2))
# ============================================================================
def sphere_sdf(x, center, radius):
    """Signed distance to a sphere/circle (negative = inside)."""
    diff = x - center
    dist = np.linalg.norm(diff)
    d = dist - radius
    grad = diff / (dist + 1e-9)         # grad d (unit vector pointing outward)
    return d, grad


def c_cost(d, eps):
    """
    CHOMP workspace cost (3-piece, C1-continuous):
        d < 0      : -d + eps/2          (inside obstacle -> strong push out)
        0 <= d<eps : (d - eps)^2 / (2eps) (inside the margin)
        d >= eps   : 0                   (safe zone)
    """
    if d < 0.0:
        return -d + 0.5 * eps
    elif d < eps:
        return 0.5 * (d - eps) ** 2 / eps
    else:
        return 0.0


def c_grad(d, eps):
    """Derivative dc/dd of the cost above."""
    if d < 0.0:
        return -1.0
    elif d < eps:
        return (d - eps) / eps
    else:
        return 0.0


def obstacle_gradient(traj, start, goal, dt, center, radius, eps):
    """
    Obstacle functional gradient (proper CHOMP form), waypoint by waypoint:

        grad F_obs(t) = ||v|| * [ (I - v_hat v_hat^T) grad_c  -  c * kappa ]

    - ||v||             : velocity weighting (invariance to time-reparametrization)
    - (I - v_hat v_hat^T): keep only the component PERPENDICULAR to velocity
                           (moving along the path does not change the cost)
    - kappa = (I - v_hat v_hat^T) a / ||v||^2 : curvature vector (resists bending)

    Returns an (n, D) array and the total cost F_obs (for logging).
    """
    n, D = traj.shape
    Q = np.vstack([start, traj, goal])     # augmented trajectory (n+2, D)
    grad = np.zeros((n, D))
    F_obs = 0.0
    I = np.eye(D)

    for i in range(n):
        x = Q[i + 1]
        v = (Q[i + 2] - Q[i]) / (2.0 * dt)              # centered velocity
        a = (Q[i + 2] - 2.0 * Q[i + 1] + Q[i]) / dt ** 2  # acceleration
        speed = np.linalg.norm(v)

        d, gd = sphere_sdf(x, center, radius)
        c   = c_cost(d, eps)
        gc  = c_grad(d, eps) * gd                        # grad_c in workspace

        F_obs += c * speed * dt

        if speed < 1e-6:                                 # guard
            grad[i] = gc
            continue

        vhat = v / speed
        P = I - np.outer(vhat, vhat)                     # projector perp. to velocity
        kappa = (P @ a) / (speed ** 2)                   # curvature
        grad[i] = speed * (P @ gc - c * kappa)

    return grad, F_obs


# ============================================================================
# 3. CHOMP LOOP: covariant descent  (block (3))
# ============================================================================
def chomp(start, goal, center, radius,
          n=60, dt=1.0, eps=2.0, lam=1.0,
          step=0.02, iters=500, tol=1e-6, verbose=False):
    """
    Optimize a start->goal trajectory that goes around a sphere.

    step  = learning rate (with preconditioner, ~0.01-0.05 works well)
    lam   = smoothness vs obstacle weight
    eps   = activation margin of the obstacle cost
    """
    start = np.asarray(start, float)
    goal  = np.asarray(goal,  float)
    D = start.size

    # --- precompute (once) ---
    A, K_free, K_bound = build_smoothness(n, dt)
    A_inv = np.linalg.inv(A + 1e-9 * np.eye(n))   # preconditioner (A^T A)^-1
    endpoints = np.vstack([start, goal])          # (2, D)
    b = (K_free.T @ K_bound) @ endpoints          # boundary offset (n, D)

    # --- initialization: straight line ---
    traj = np.linspace(start, goal, n + 2)[1:-1].copy()   # (n, D)
    init = traj.copy()

    history = []
    for it in range(iters):
        g_smooth = A @ traj + b
        g_obs, F_obs = obstacle_gradient(traj, start, goal, dt,
                                         center, radius, eps)
        g = lam * g_smooth + g_obs

        # ---- COVARIANT UPDATE ----
        traj = traj - step * (A_inv @ g)

        accel = K_free @ traj + K_bound @ endpoints
        F_smooth = 0.5 * np.sum(accel ** 2)
        U = lam * F_smooth + F_obs
        history.append(U)

        if verbose and it % 50 == 0:
            print(f"  it {it:3d} | U = {U:10.4f} | F_obs = {F_obs:8.4f}")

        if it > 0 and abs(history[-2] - history[-1]) < tol:
            break

    return traj, init, np.array(history)


# ============================================================================
# 4. VALIDATION METRICS
# ============================================================================
def min_clearance(traj, start, goal, center, radius):
    """Minimum distance to the obstacle surface (>0 = collision-free)."""
    Q = np.vstack([start, traj, goal])
    dists = [np.linalg.norm(x - center) - radius for x in Q]
    return min(dists)


def path_length(traj, start, goal):
    Q = np.vstack([start, traj, goal])
    return np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1))


# ============================================================================
# 5. DEMO (Experiment 1): one obstacle right on the straight line
# ============================================================================
if __name__ == "__main__":
    start  = np.array([0.0, 0.0])
    goal   = np.array([10.0, 0.0])
    center = np.array([5.0, 0.6])      # slightly offset obstacle
    radius = 2.0

    print("CHOMP single agent - demo")
    traj, init, hist = chomp(start, goal, center, radius,
                             n=60, dt=1.0, eps=1.5, lam=0.6,
                             step=0.02, iters=500, verbose=True)

    clr_i = min_clearance(init, start, goal, center, radius)
    clr_f = min_clearance(traj, start, goal, center, radius)
    len_i = path_length(init, start, goal)
    len_f = path_length(traj, start, goal)
    print("\n--- Validation ---")
    print(f"Clearance  init : {clr_i:+.3f}  (negative = crosses the obstacle)")
    print(f"Clearance  final: {clr_f:+.3f}  (positive = collision-free)")
    print(f"Length     init : {len_i:.3f}")
    print(f"Length     final: {len_f:.3f}  (+{100*(len_f-len_i)/len_i:.1f}% to avoid)")
    print(f"Cost U     init : {hist[0]:.3f}  ->  final: {hist[-1]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.fill(center[0] + radius * np.cos(th),
             center[1] + radius * np.sin(th),
             color="#d96459", alpha=0.35, label="obstacle")
    ax1.plot(center[0] + radius * np.cos(th),
             center[1] + radius * np.sin(th), color="#d96459", lw=1.5)
    Qi = np.vstack([start, init, goal])
    Qf = np.vstack([start, traj, goal])
    ax1.plot(Qi[:, 0], Qi[:, 1], "--", color="#888", lw=1.8, label="init (straight line)")
    ax1.plot(Qf[:, 0], Qf[:, 1], "-", color="#2a7", lw=2.5, label="CHOMP (covariant)")
    ax1.scatter(*start, c="k", zorder=5); ax1.scatter(*goal, c="k", zorder=5)
    ax1.set_aspect("equal"); ax1.legend(); ax1.set_title("Trajectory")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.grid(alpha=0.3)

    ax2.plot(hist, color="#27a", lw=2)
    ax2.set_yscale("log")
    ax2.set_title("Cost convergence U(xi)")
    ax2.set_xlabel("iteration"); ax2.set_ylabel("U (log scale)")
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("COSMOS - Part I: CHOMP single agent (covariant gradient)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("chomp_part1_demo.png", dpi=130)
    print("\nFigure -> chomp_part1_demo.png")
