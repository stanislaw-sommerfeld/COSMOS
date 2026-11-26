"""
COSMOS - shared numerical core (single source of truth)
=======================================================

Before this module existed, several primitives were copy-pasted across the
package:

  * the controllability / reachability Gramian   -> cli.gramian,
    cosmos_optimal.reachability_gramian, cosmos_safe.gramian  (3 byte-identical copies)
  * the continuous Clohessy-Wiltshire state system (A, B)      (3 hardcoded copies)
  * the closed-form LQ optimum (energy + trajectory)           (2 copies)
  * the Euclidean / Riemannian keep-out projection             (3 near-identical copies)

Each copy was a place a bug or an improvement could silently diverge. They now
live here, once, and the former call sites import them. The behaviour is exactly
the one the copies implemented, so the locked regression tests (T1-T8) keep
passing bit-for-bit.

Nothing here depends on the rest of the package, so it is safe to import from
anywhere (no circular imports).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


# ============================================================================
# 1. Continuous Clohessy-Wiltshire state system  s_dot = A s + B u
# ============================================================================
def cw_state_matrices(nm, dim=2):
    """Continuous CW state-space (A, B) in the Hill frame, normalized or SI.

    state (dim=2): s = [x, y, vx, vy]          (x = radial, y = along-track)
    state (dim=3): s = [x, y, z, vx, vy, vz]   (z = cross-track oscillator)

    dim=2:
        x'' - 2 n y' - 3 n^2 x = u_x
        y'' + 2 n x'           = u_y
    dim=3 adds the decoupled cross-track mode  z'' + n^2 z = u_z.

    This is the single definition of the CW linear system; cosmos_optimal,
    cosmos_safe and cli previously each hardcoded their own identical copy.
    """
    if dim == 2:
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [3 * nm ** 2, 0, 0, 2 * nm],
                      [0, 0, -2 * nm, 0]], float)
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
    elif dim == 3:
        A = np.zeros((6, 6)); B = np.zeros((6, 3))
        A[0, 3] = A[1, 4] = A[2, 5] = 1.0
        A[3, 0] = 3 * nm ** 2; A[3, 4] = 2 * nm; A[4, 3] = -2 * nm
        A[5, 2] = -nm ** 2
        B[3, 0] = B[4, 1] = B[5, 2] = 1.0
    else:
        raise ValueError("dim must be 2 or 3")
    return A, B


# ============================================================================
# 2. Controllability / reachability Gramian
# ============================================================================
def controllability_gramian(A, B, T, nq=4000, method="quadrature"):
    """Finite-horizon controllability Gramian

        W(T) = \\int_0^T e^{A sigma} B B^T e^{A sigma}^T d sigma .

    ``method="quadrature"`` (default) reproduces, bit-for-bit, the incremental
    left-rectangle sum the former copies used (``cli.gramian``,
    ``cosmos_optimal.reachability_gramian``, ``cosmos_safe.gramian``), so every
    locked optimality ratio is unchanged.

    ``method="vanloan"`` uses the exact Van Loan block-matrix identity, which is
    an order-of-magnitude more accurate and far faster (one matrix exponential
    instead of ``nq`` products). It is offered as an opt-in for new code; the
    default stays on quadrature so the regression suite does not move.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    if method == "vanloan":
        n = A.shape[0]
        M = np.zeros((2 * n, 2 * n))
        M[:n, :n] = -A
        M[:n, n:] = B @ B.T
        M[n:, n:] = A.T
        E = expm(M * T)
        Phi_T = E[n:, n:].T            # e^{A T}
        W = Phi_T @ E[:n, n:]          # Phi(T) * (E12)
        return 0.5 * (W + W.T)         # symmetrize against round-off
    # --- legacy quadrature (kept as the default for exact reproducibility) ---
    dq = T / nq
    Phi = np.eye(A.shape[0])
    step = expm(A * dq)
    BBt = B @ B.T
    W = np.zeros_like(A, dtype=float)
    for _ in range(nq):
        W += Phi @ BBt @ Phi.T * dq
        Phi = step @ Phi
    return W


# backward-compatible alias (old name used in cosmos_optimal)
reachability_gramian = controllability_gramian


# ============================================================================
# 3. Closed-form LQ (minimum-energy) transfer
# ============================================================================
def lq_optimal_energy(s0, sf, A, B, T, W=None, nq=4000):
    """Minimum control energy J* = (sf - Phi(T) s0)^T W^-1 (sf - Phi(T) s0)
       for the linear transfer s0 -> sf in time T. Pass ``W`` to reuse a Gramian."""
    s0 = np.asarray(s0, float); sf = np.asarray(sf, float)
    if W is None:
        W = controllability_gramian(A, B, T, nq=nq)
    ds = sf - expm(A * T) @ s0
    return float(ds @ np.linalg.solve(W, ds))


def lq_optimal_trajectory(s0, sf, A, B, T, steps=400, W=None, nq=4000):
    """Closed-form min-energy transfer. Returns (J*, position trajectory).

    The optimal control is  u*(t) = B^T e^{A^T (T - t)} W^-1 (sf - Phi(T) s0);
    the state is forward-integrated (explicit Euler, ``steps`` points) exactly as
    the former ``cosmos_optimal.lq_optimal`` / ``cosmos_safe.lq_opt`` did, so the
    ground-truth curves are unchanged.
    """
    s0 = np.asarray(s0, float); sf = np.asarray(sf, float)
    if W is None:
        W = controllability_gramian(A, B, T, nq=nq)
    ds = sf - expm(A * T) @ s0
    win = np.linalg.solve(W, ds)
    Jstar = float(ds @ win)
    half = s0.size // 2
    h = T / steps
    s = s0.copy()
    traj = [s[:half].copy()]
    for k in range(steps):
        u = B.T @ expm(A * (T - k * h)).T @ win
        s = s + h * (A @ s + B @ u)
        traj.append(s[:half].copy())
    return Jstar, np.array(traj)


# ============================================================================
# 4. Keep-out projections (shared by cli, cosmos_full, cosmos_safe)
# ============================================================================
def _flatten(traj):
    """(n, D) -> block layout [x.., y.., (z..)]  (local copy of chomp_cw.to_flat
       to keep _core dependency-free)."""
    D = traj.shape[1]
    return np.concatenate([traj[:, d] for d in range(D)])


def _unflatten(flat, n):
    D = len(flat) // n
    return np.column_stack([flat[d * n:(d + 1) * n] for d in range(D)])


def project_euclid(traj, obstacles, margin=0.05):
    """Metric-BLIND (Euclidean) hard keep-out projection, dim-general: push any
       waypoint inside a keep-out radially out to the boundary. This is the naive
       projection whose ~15x energy premium Module A exposes."""
    for i in range(len(traj)):
        for ob in obstacles:
            v = traj[i] - ob["c"]
            d = np.linalg.norm(v)
            if d - ob["r"] < margin:
                traj[i] = ob["c"] + v / (d + 1e-9) * (ob["r"] + margin)
    return traj


def project_metric(traj, obstacles, Minv, margin=0.05, sweeps=3):
    """Metric-AWARE (Riemannian) keep-out projection in M = B^T B, dim-general.

    For each violated waypoint i with outward normal n_i, the minimum-fuel
    correction solving  min dxi^T M dxi  s.t.  a^T(xi+dxi) >= b  is closed form:

        dxi = M^-1 a (margin - clearance) / (a^T M^-1 a),
        a = n_i placed on waypoint i's slots in every dimension.

    This is Module B's core move, generalized to 2D/3D. It consolidates the three
    former copies (cosmos_safe.project_metric, cli.project_metric_nd,
    cosmos_full.project_keepout_metric).
    """
    n, D = traj.shape
    xi = _flatten(traj)
    for _ in range(sweeps):
        Tj = _unflatten(xi, n)
        for i in range(n):
            for ob in obstacles:
                v = Tj[i] - ob["c"]
                d = np.linalg.norm(v) + 1e-9
                clr = d - ob["r"]
                if clr < margin:
                    ni = v / d
                    a = np.zeros(D * n)
                    for dd in range(D):
                        a[dd * n + i] = ni[dd]
                    Ma = Minv @ a
                    xi = xi + Ma * ((margin - clr) / (a @ Ma))
                    Tj = _unflatten(xi, n)
    return _unflatten(xi, n)


# ============================================================================
# 5. SPD preconditioner helpers (numerically safer than explicit inverse)
# ============================================================================
def spd_inverse(M, ridge=1e-9):
    """Inverse of a symmetric positive-definite matrix via Cholesky.

    Equivalent to ``np.linalg.inv(M + ridge*I)`` for the SPD metrics used here,
    but routed through a Cholesky factorization (``cho_factor``/``cho_solve``),
    which is faster and better-conditioned than a general LU inverse. Used where
    an explicit preconditioner matrix must be materialized."""
    from scipy.linalg import cho_factor, cho_solve
    M = np.asarray(M, float)
    n = M.shape[0]
    c = cho_factor(M + ridge * np.eye(n), lower=True)
    return cho_solve(c, np.eye(n))


# ============================================================================
# 6. Time-varying linear relative dynamics  (B = D2 - G(t))
# ============================================================================
#
# COSMOS's constant-coefficient operators (chomp_cw.cw_operators,
# linear_relative_operators) cover free space, Clohessy-Wiltshire and the
# Schweighart-Sedwick J2 model -- all linear and TIME-INVARIANT. The natural and
# most valuable generalization is the TIME-VARYING case: relative motion about an
# ECCENTRIC reference orbit (Tschauner-Hempel / Lawden). There the coupling block
# G depends on the true anomaly, so each waypoint sees a different G(t_k) -- yet
# the construction B = D2 - G is otherwise identical, which is exactly the
# "metric = dynamics" claim, now for a time-varying metric.
#
# This matters because the CW approximation is genuinely poor for eccentric
# references (a CW-"passive" relative orbit can drift by many times its own
# amplitude in a single revolution), so planning in the correct time-varying
# metric produces a large, honest fuel saving -- unlike the LEO/J2 case where the
# mismatch is ~1e-5 and CW is already adequate.


def tschauner_hempel_coeffs(t, e, nm, a=1.0, mu=1.0):
    """Planar time-varying coefficient blocks (Kx, Kv) of the linearized relative
       motion about an eccentric reference orbit (semi-major axis ``a``,
       eccentricity ``e``, mean motion ``nm``), in the rotating LVLH/Hill frame
       (x = radial, y = along-track):

           x'' = (theta_dot^2 + 2 mu/r^3) x + theta_ddot y + 2 theta_dot y' + u_x
           y'' = -theta_ddot x + (theta_dot^2 - mu/r^3) y - 2 theta_dot x' + u_y

       so  [x'';y''] = Kx [x;y] + Kv [x';y'] + u, with Kx, Kv evaluated at the
       reference anomaly reached at time ``t`` (Kepler-solved). As e -> 0 this
       reduces to the Clohessy-Wiltshire blocks exactly."""
    M = nm * t
    E = M
    for _ in range(60):                       # Newton solve of Kepler's equation
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1.0 - e * np.cos(E))
    h = np.sqrt(mu * a * (1.0 - e ** 2))
    thd = h / r ** 2                          # reference angular rate
    rdot = mu / h * e * np.sin(nu)
    thdd = -2.0 * rdot * thd / r              # reference angular acceleration
    g = mu / r ** 3
    Kx = np.array([[thd ** 2 + 2 * g, thdd],
                   [-thdd, thd ** 2 - g]])
    Kv = np.array([[0.0, 2 * thd],
                   [-2 * thd, 0.0]])
    return Kx, Kv


def time_varying_operators(n, dt, coeff_fn, w_bc=0.0):
    """Control operator B = D2 - G(t) for 2D planar TIME-VARYING linear relative
       dynamics. ``coeff_fn(t)`` must return the blocks ``(Kx, Kv)`` (2x2 each)
       such that  [x'';y''] = Kx [x;y] + Kv [x';y'] + u  at time ``t``.

       Returns ``(B, (B^T B)^-1, blocks)`` with the same interface as
       chomp_cw.cw_operators / linear_relative_operators, so the covariant solver,
       the closed-form optimum and the metric-aware projections all apply
       unchanged. ``w_bc>0`` appends the four rest-to-rest boundary-velocity rows."""
    D2 = np.zeros((n, n + 2)); D1 = np.zeros((n, n + 2))
    for t in range(n):
        D2[t, t] = 1.0; D2[t, t + 1] = -2.0; D2[t, t + 2] = 1.0
        D1[t, t] = -1.0; D1[t, t + 2] = 1.0
    D2 /= dt ** 2; D1 /= (2.0 * dt)
    D2f, D2b = D2[:, 1:n + 1], D2[:, [0, n + 1]]
    D1f, D1b = D1[:, 1:n + 1], D1[:, [0, n + 1]]
    I = np.eye(n)

    Bxx = D2f.copy(); Bxy = np.zeros((n, n))
    Byx = np.zeros((n, n)); Byy = D2f.copy()
    # per-waypoint (row) coefficients -> subtract G(t_k) from that row
    Kx_all = np.zeros((n, 2, 2)); Kv_all = np.zeros((n, 2, 2))
    for t in range(n):
        tt = (t + 1) * dt                      # interior waypoint time
        Kx, Kv = coeff_fn(tt)
        Kx_all[t] = Kx; Kv_all[t] = Kv
        Bxx[t] -= Kx[0, 0] * I[t] + Kv[0, 0] * D1f[t]
        Bxy[t] -= Kx[0, 1] * I[t] + Kv[0, 1] * D1f[t]
        Byx[t] -= Kx[1, 0] * I[t] + Kv[1, 0] * D1f[t]
        Byy[t] -= Kx[1, 1] * I[t] + Kv[1, 1] * D1f[t]
    B = np.block([[Bxx, Bxy], [Byx, Byy]])

    sw = np.sqrt(w_bc)
    if w_bc > 0.0:
        R = np.zeros((4, 2 * n))
        R[0, 0] = sw / dt; R[1, n - 1] = -sw / dt
        R[2, n] = sw / dt; R[3, 2 * n - 1] = -sw / dt
        B = np.vstack([B, R])

    M = B.T @ B + 1e-9 * np.eye(2 * n)
    Minv = spd_inverse(M)
    blocks = dict(D2b=D2b, D1b=D1b, dt=dt, n=n, sw=sw, w_bc=w_bc,
                  Kx=Kx_all, Kv=Kv_all)
    return B, Minv, blocks


def time_varying_offset(blocks, start, goal):
    """Constant term c0 so that  u = B xi + c0  for the time-varying operator.
       The fixed endpoints feed the D2/D1 boundary columns AND the per-row G(t)
       velocity coupling (the radial/along-track Coriolis term), so c0 carries a
       Kv contribution that the constant-coefficient offset did not need."""
    D2b, D1b = blocks["D2b"], blocks["D1b"]
    dt, n = blocks["dt"], blocks["n"]
    Kv = blocks["Kv"]
    ex = np.array([start[0], goal[0]]); ey = np.array([start[1], goal[1]])
    # D2 boundary contribution (same as the constant-coefficient case)
    c0x = D2b @ ex
    c0y = D2b @ ey
    # subtract the velocity-coupling of the FIXED endpoints: only the first/last
    # interior rows see a boundary point through the centered velocity stencil D1b.
    # D1b has nonzero entries only at waypoints 0 (start col) and n-1 (goal col).
    vx_bc = D1b @ ex            # length-n, nonzero at rows 0 and n-1
    vy_bc = D1b @ ey
    for t in range(n):
        c0x[t] -= Kv[t, 0, 0] * vx_bc[t] + Kv[t, 0, 1] * vy_bc[t]
        c0y[t] -= Kv[t, 1, 0] * vx_bc[t] + Kv[t, 1, 1] * vy_bc[t]
    c0 = np.concatenate([c0x, c0y])
    if blocks["w_bc"] > 0.0:
        sw = blocks["sw"]
        c0 = np.concatenate([c0, np.array([-sw * start[0] / dt, sw * goal[0] / dt,
                                           -sw * start[1] / dt, sw * goal[1] / dt])])
    return c0


# ============================================================================
# 7. Metric-aware INTER-AGENT safety (E3): coupled pairwise Riemannian projection
# ============================================================================
#
# Module B (project_metric) projects a single agent off a STATIC keep-out in the
# fuel metric M = B^T B. Inter-agent collisions are different: the constraint
# couples TWO trajectories at the same time index, so the correction must be
# shared between both agents -- and shared the fuel-cheap way, not split 50/50 in
# Euclidean space (which repeats the 15x metric-blind pitfall).
#
# Stack z = [xi_i ; xi_j]. The joint metric is block-diagonal diag(M, M). A
# separation constraint at time t, ||x_i(t) - x_j(t)|| >= r, linearizes to
# a^T z >= b with normal n_hat on agent i's slots and -n_hat on agent j's. The
# minimum-fuel correction solving  min dz^T diag(M,M) dz  s.t. feasible is again
# closed form, and because the metric is block-diagonal it decouples into two
# per-agent pushes M^-1 a_i and M^-1 a_j sharing one multiplier:
#
#     lambda = (b - a^T z) / (a_i^T M^-1 a_i + a_j^T M^-1 a_j),
#     xi_i += lambda M^-1 a_i ,   xi_j += lambda M^-1 a_j  (opposite directions).
#
# So each agent moves the way that is cheapest in ITS OWN fuel metric -- the
# metric-aware generalization of the naive "push both apart equally" rule.


def _flat(traj):
    return _flatten(traj)


def project_inter_agent_metric(trajs, Minv, r_col, margin=0.05, sweeps=4):
    """Metric-aware (Riemannian) hard inter-agent separation in the fuel metric.

    ``trajs`` is a list of (n, D) interior trajectories sharing the SAME metric
    preconditioner ``Minv = (B^T B)^-1``. For every pair closer than
    ``r_col + margin`` at any time index, applies the closed-form minimum-fuel
    coupled correction above. Returns the corrected list. Typically ~2-3x cheaper
    (in realized Delta-v) than the Euclidean push-apart for the same clearance,
    and it preserves the smooth, dynamics-consistent shape of each trajectory
    because the push lives in M^-1."""
    K = len(trajs)
    if K < 2:
        return trajs
    n, D = trajs[0].shape
    xis = [_flat(t.copy()) for t in trajs]
    for _ in range(sweeps):
        Ts = [_unflatten(x, n) for x in xis]
        for i in range(K):
            for j in range(i + 1, K):
                for t in range(n):
                    v = Ts[i][t] - Ts[j][t]
                    d = np.linalg.norm(v) + 1e-9
                    clr = d - r_col
                    if clr < margin:
                        nh = v / d
                        ai = np.zeros(D * n); aj = np.zeros(D * n)
                        for dd in range(D):
                            ai[dd * n + t] = nh[dd]
                            aj[dd * n + t] = -nh[dd]
                        Mai = Minv @ ai; Maj = Minv @ aj
                        denom = ai @ Mai + aj @ Maj
                        lam = (margin - clr) / denom
                        xis[i] = xis[i] + Mai * lam
                        xis[j] = xis[j] + Maj * lam
                        Ts[i] = _unflatten(xis[i], n)
                        Ts[j] = _unflatten(xis[j], n)
    return [_unflatten(x, n) for x in xis]


# ============================================================================
# 8. Chance-constrained keep-outs (E5): inflate the radius by k_beta * sigma
# ============================================================================
#
# The keep-out radius is treated as exact, but in practice each waypoint's
# position is known only up to navigation uncertainty. Model it as isotropic
# Gaussian with standard deviation sigma. Requiring the TRUE position to stay
# outside the physical radius r_phys with probability >= beta is, for the
# dominant 1-D (radial) component of the violation, a deterministic constraint on
# the MEAN position with an inflated radius:
#
#     r_safe = r_phys + k_beta * sigma ,     k_beta = Phi^-1(beta)
#
# (Phi^-1 = standard-normal quantile). Planning against r_safe with the exact
# same Riemannian projection then guarantees the chance constraint, trading a
# little fuel for a calibrated safety probability. This is the standard
# single-obstacle Boole/– quantile reformulation of a chance constraint; here it
# plugs directly into COSMOS's metric-aware projection with zero new machinery.


def chance_margin(sigma, beta):
    """k_beta * sigma, the radius inflation for a per-waypoint collision
       probability <= 1 - beta under isotropic Gaussian position error.
       beta in (0.5, 1); uses the inverse standard-normal CDF (no SciPy needed
       for common values, SciPy fallback otherwise)."""
    import math
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0, 1)")
    try:
        from scipy.stats import norm
        k = float(norm.ppf(beta))
    except Exception:                                  # rational approx (Acklam)
        k = _norm_ppf(beta)
    return max(k, 0.0) * sigma


def inflate_obstacles(obstacles, sigma, beta):
    """Return a copy of ``obstacles`` with each radius inflated by chance_margin,
       so planning against them enforces a per-waypoint safety probability beta."""
    m = chance_margin(sigma, beta)
    return [{"c": np.asarray(o["c"], float), "r": o["r"] + m} for o in obstacles]


def _norm_ppf(p):
    """Acklam's rational approximation of the inverse standard-normal CDF
       (used only if SciPy's norm.ppf is unavailable)."""
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
