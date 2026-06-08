"""
COSMOS - regression tests that lock the headline claims.
========================================================

Run with:  pytest -q     (from the repo root, after `pip install -e .`)

Each test turns one README/report claim into a guarantee that is re-checked on every
commit, so a refactor cannot silently break a result:

  T1  n -> 0 reduces COSMOS to free-space CHOMP  (B -> block-diag D2): the fuel
      metric must collapse onto the raw-acceleration metric.
  T2  a free-space plan scored under n=0 spends ~0 delta-v (the straight line is
      already a geodesic) -- the project's own sanity check.
  T3  fuel-aware planning beats dynamics-blind planning under the TRUE dynamics.
  T4  fuel-aware CHOMP recovers the analytical LQ / Gramian optimum to ~1.00x
      (Module A) -- the optimality claim.
  T5  covariant gradient descent (no obstacle) converges to the closed-form
      unconstrained optimum -- i.e. the headline number is converged, not tuned.
  T6  the full mission (cli.solve) is collision-free with the Riemannian safety.
"""

import numpy as np
import pytest

from cosmos.chomp_cw import (cw_operators, chomp_cw, delta_v,
                             cw_unconstrained_optimum, to_flat)


# ---------------------------------------------------------------------------
# T1 - n->0 collapses the fuel metric onto the free-space (D2) metric
# ---------------------------------------------------------------------------
def test_n_to_zero_recovers_free_space():
    N, dt = 40, 0.05
    B0, _, _ = cw_operators(N, dt, nm=0.0, dim=2)
    # at n=0 the in-plane coupling vanishes -> B is block-diagonal [D2, D2]
    n = N
    off_diag_xy = B0[:n, n:]      # radial rows, along-track cols
    off_diag_yx = B0[n:, :n]
    assert np.allclose(off_diag_xy, 0.0, atol=1e-12)
    assert np.allclose(off_diag_yx, 0.0, atol=1e-12)
    # and the fuel metric -> free-space CONTINUOUSLY as n -> 0 (no discontinuity):
    # the deviation from B0 must shrink as the mean motion shrinks.
    d_small = np.linalg.norm(cw_operators(N, dt, nm=1e-5, dim=2)[0] - B0)
    d_large = np.linalg.norm(cw_operators(N, dt, nm=1e-3, dim=2)[0] - B0)
    assert d_small < d_large
    assert d_small < 1e-3 * np.linalg.norm(B0)


# ---------------------------------------------------------------------------
# T2 - free-space plan scored at n=0 spends ~0 delta-v (the straight line)
# ---------------------------------------------------------------------------
def test_straight_line_costs_nothing_at_zero_n():
    N = 60
    T = np.pi
    dt = T / (N + 1)
    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    traj, _, _ = chomp_cw(start, goal, nm=0.0, n=N, dt=dt, iters=200)
    dv, _ = delta_v(traj, start, goal, dt, nm=0.0)
    assert dv < 1e-3


# ---------------------------------------------------------------------------
# T3 - fuel-aware beats dynamics-blind under the true dynamics
# ---------------------------------------------------------------------------
def test_fuel_aware_beats_free_space():
    N = 60
    T = np.pi
    dt = T / (N + 1)
    nm_true = 1.0
    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    t_free, _, _ = chomp_cw(start, goal, nm=0.0, n=N, dt=dt, iters=400)
    t_fuel, _, _ = chomp_cw(start, goal, nm=nm_true, n=N, dt=dt, iters=400)
    dv_free, _ = delta_v(t_free, start, goal, dt, nm_true)
    dv_fuel, _ = delta_v(t_fuel, start, goal, dt, nm_true)
    assert dv_fuel < dv_free


# ---------------------------------------------------------------------------
# T4 - fuel-aware CHOMP recovers the analytical LQ optimum (Module A, ~1.00x)
# ---------------------------------------------------------------------------
def test_recovers_lq_optimum():
    from scipy.linalg import expm
    nm = 1.0
    T = np.pi
    N = 60
    dt = T / (N + 1)
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                  [3 * nm ** 2, 0, 0, 2 * nm], [0, 0, -2 * nm, 0]], float)
    Bm = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)

    # analytical min-energy transfer via the reachability Gramian
    nq = 4000
    dq = T / nq
    Phi = np.eye(4)
    stepm = expm(A * dq)
    BBt = Bm @ Bm.T
    W = np.zeros((4, 4))
    for _ in range(nq):
        W += Phi @ BBt @ Phi.T * dq
        Phi = stepm @ Phi
    s0 = np.array([3.0, 5.0, 0.0, 0.0])
    sf = np.zeros(4)
    ds = sf - expm(A * T) @ s0
    Jstar = ds @ np.linalg.solve(W, ds)

    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    t_fuel, _, _ = chomp_cw(start, goal, nm=nm, n=N, dt=dt, w_bc=20000.0,
                            step=0.02, iters=2000)
    _, E_fuel = delta_v(t_fuel, start, goal, dt, nm)
    ratio = E_fuel / Jstar
    # must be at the optimum, not below it (can't beat the analytical optimum),
    # and not far above it.
    assert 0.97 <= ratio <= 1.08, f"energy ratio vs LQ optimum = {ratio:.3f}"


# ---------------------------------------------------------------------------
# T5 - descent (no obstacle) converges to the closed-form optimum
# ---------------------------------------------------------------------------
def test_descent_reaches_closed_form_optimum():
    N = 60
    T = np.pi
    dt = T / (N + 1)
    nm = 1.0
    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    t_gd, _, _ = chomp_cw(start, goal, nm=nm, n=N, dt=dt, step=0.02, iters=4000)
    t_cf = cw_unconstrained_optimum(start, goal, nm=nm, n=N, dt=dt)
    rel = np.linalg.norm(to_flat(t_gd) - to_flat(t_cf)) / np.linalg.norm(to_flat(t_cf))
    assert rel < 0.02, f"gradient descent off the closed-form optimum by {rel:.3%}"


# ---------------------------------------------------------------------------
# T6 - the full mission is collision-free with Riemannian safety
# ---------------------------------------------------------------------------
def test_mission_is_collision_free():
    from cosmos import cli
    starts, goals0, obstacles, _ = cli.scenario("reconfig", dim=2)
    goals, _ = cli.assign(starts, goals0, "hungarian", 1.0, np.pi, 2)
    N = 50
    dt = np.pi / (N + 1)
    trajs = cli.solve(starts, goals, nm=1.0, n=N, dt=dt, dim=2,
                      obstacles=obstacles, safety="riemann", iters=600)
    clr = cli.min_pair_clearance(trajs, starts, goals)
    assert clr > 0.0, f"min inter-agent clearance = {clr:.3f} (collision)"


# ---------------------------------------------------------------------------
# T7 - the generality claim: CW is the special case of B = D2 - G
# ---------------------------------------------------------------------------
def test_cw_is_special_case_of_general_dynamics():
    from cosmos.chomp_cw import (cw_operators, linear_relative_operators,
                                 cw_coeffs, ss_coeffs)
    N, dt, nm = 40, 0.05, 1.0
    B_cw, _, _ = cw_operators(N, dt, nm, w_bc=0.0, dim=2)
    kxx, kc = cw_coeffs(nm)
    B_gen, _, _ = linear_relative_operators(N, dt, kxx, kc, w_bc=0.0)
    # plugging the CW coefficients into the generic builder reproduces cw_operators
    assert np.allclose(B_cw, B_gen, atol=1e-10)
    # J2/SS gives a DIFFERENT but well-formed operator that -> CW as J2 -> 0
    kxx2, kc2 = ss_coeffs(nm, np.deg2rad(51.6))
    assert kxx2 != kxx and kc2 != kc          # genuinely different dynamics
    assert abs(kxx2 - kxx) < 0.1 and abs(kc2 - kc) < 0.1   # but close (small J2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))