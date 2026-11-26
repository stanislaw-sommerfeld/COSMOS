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
