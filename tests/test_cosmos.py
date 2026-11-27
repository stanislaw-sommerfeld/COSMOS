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
# T8 - line-search reaches the closed-form optimum tightly, in a few iterations
# ---------------------------------------------------------------------------
def test_linesearch_converges_fast_and_exactly():
    N = 60
    T = np.pi
    dt = T / (N + 1)
    nm = 1.0
    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    t_ls, _, hist = chomp_cw(start, goal, nm=nm, n=N, dt=dt)   # line-search on by default
    t_cf = cw_unconstrained_optimum(start, goal, nm=nm, n=N, dt=dt)
    rel = np.linalg.norm(to_flat(t_ls) - to_flat(t_cf)) / np.linalg.norm(to_flat(t_cf))
    assert rel < 1e-3, f"line-search off the closed-form optimum by {rel:.3%}"
    assert len(hist) < 25, f"line-search took {len(hist)} iterations (expected a handful)"


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


# ---------------------------------------------------------------------------
# T9 - time-varying (Tschauner-Hempel) metric reduces to CW at e=0
# ---------------------------------------------------------------------------
def test_time_varying_metric_reduces_to_cw_at_zero_eccentricity():
    from cosmos._core import (time_varying_operators, time_varying_offset,
                              tschauner_hempel_coeffs)
    N, T, nm = 60, np.pi, 1.0
    dt = T / (N + 1)
    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    coeff = lambda t: tschauner_hempel_coeffs(t, e=0.0, nm=nm)
    B_tv, _, bl = time_varying_operators(N, dt, coeff, w_bc=0.0)
    B_cw, _, blcw = cw_operators(N, dt, nm, w_bc=0.0, dim=2)
    # the time-varying construction is a strict superset: e=0 -> CW, bit-for-bit
    assert np.allclose(B_tv, B_cw, atol=1e-12)
    from cosmos.chomp_cw import cw_offset
    assert np.allclose(time_varying_offset(bl, start, goal),
                       cw_offset(blcw, start, goal), atol=1e-12)


# ---------------------------------------------------------------------------
# T10 - on an eccentric orbit, the TH metric genuinely beats the CW metric
# ---------------------------------------------------------------------------
def test_eccentric_metric_beats_cw_metric():
    from cosmos import cosmos_eccentric as ecc
    rows = ecc.run(eccentricities=(0.0, 0.3), N=70, verbose=False)
    by_e = {r["e"]: r for r in rows}
    # circular (e=0): the CW metric is exactly right -> saving ~ 0
    assert abs(by_e[0.0]["saving"]) < 0.5, f"e=0 saving {by_e[0.0]['saving']:.2f}% (should be ~0)"
    # eccentric (e=0.3): the time-varying metric saves a real, large fraction
    assert by_e[0.3]["saving"] > 5.0, f"e=0.3 saving {by_e[0.3]['saving']:.2f}% (should be large)"


# ---------------------------------------------------------------------------
# T10b - the eccentric result also holds in 3D (cross-track), scored under the
#        full 3D nonlinear dynamics; and the 3D TH operator reduces to CW-3D at e=0
# ---------------------------------------------------------------------------
def test_eccentric_metric_holds_in_3d():
    from cosmos._core import (time_varying_operators, time_varying_offset,
                              tschauner_hempel_coeffs)
    from cosmos.chomp_cw import cw_operators, cw_offset
    N, T, nm = 50, np.pi, 1.0
    dt = T / (N + 1)
    # 3D at e=0 reproduces CW-3D bit-for-bit (strict superset)
    coeff = lambda t: tschauner_hempel_coeffs(t, e=0.0, nm=nm, dim=3)
    B_tv, _, bl = time_varying_operators(N, dt, coeff, w_bc=0.0, dim=3)
    B_cw, _, blcw = cw_operators(N, dt, nm, w_bc=0.0, dim=3)
    assert np.allclose(B_tv, B_cw, atol=1e-12)
    s3, g3 = np.array([3.0, 5.0, 2.0]), np.array([0.0, 0.0, 0.0])
    assert np.allclose(time_varying_offset(bl, s3, g3),
                       cw_offset(blcw, s3, g3), atol=1e-12)
    # and the saving curve in 3D behaves like 2D: ~0 at e=0, large at e=0.3
    from cosmos import cosmos_eccentric as ecc
    rows = ecc.run(eccentricities=(0.0, 0.3), N=70, dim=3, verbose=False)
    by = {r["e"]: r for r in rows}
    assert abs(by[0.0]["saving"]) < 0.5
    assert by[0.3]["saving"] > 5.0


# ---------------------------------------------------------------------------
# T11 - the covariant preconditioner is the exact Gauss-Newton step
# ---------------------------------------------------------------------------
def test_preconditioner_is_exact_gauss_newton():
    from cosmos import cosmos_newton as nwt
    N, T, nm = 60, np.pi, 1.0
    dt = T / (N + 1)
    start, goal = np.array([3.0, 5.0]), np.array([0.0, 0.0])
    # a single unit covariant step must reach the closed-form optimum to ~machine eps
    err = nwt.one_step_error(start, goal, nm, N, dt)
    assert err < 1e-6, f"one Newton step off the optimum by {err:.2e}"


# ---------------------------------------------------------------------------
# T12 - the Van Loan Gramian agrees with the legacy quadrature Gramian
# ---------------------------------------------------------------------------
def test_vanloan_gramian_matches_quadrature():
    from cosmos._core import cw_state_matrices, controllability_gramian
    A, B = cw_state_matrices(1.0, dim=2)
    Wq = controllability_gramian(A, B, np.pi, method="quadrature")
    Wv = controllability_gramian(A, B, np.pi, method="vanloan")
    rel = np.linalg.norm(Wq - Wv) / np.linalg.norm(Wv)
    assert rel < 1e-3, f"Van Loan vs quadrature Gramian differ by {rel:.2e}"
    # Van Loan is symmetric by construction
    assert np.allclose(Wv, Wv.T, atol=1e-10)


# ---------------------------------------------------------------------------
# T13 - E3: metric-aware inter-agent safety is collision-free and >= as cheap
# ---------------------------------------------------------------------------
def test_inter_agent_metric_safety_is_cheaper_and_safe():
    from cosmos import cosmos_swarm_safe as e3
    r = e3.run(verbose=False)
    # both methods achieve the required clearance (>= r_col, within the margin)
    assert r["d_ri"] >= r["r_col"] - 1e-6, f"Riemannian clearance {r['d_ri']:.3f}"
    assert r["d_eu"] >= r["r_col"] - 1e-6, f"Euclidean clearance {r['d_eu']:.3f}"
    # metric-aware separation costs no more Delta-v than the metric-blind push
    assert r["dv_ri"] <= r["dv_eu"] + 1e-6, \
        f"Riemannian dv {r['dv_ri']:.2f} should be <= Euclidean {r['dv_eu']:.2f}"


# ---------------------------------------------------------------------------
# T14 - E4: receding-horizon replanning cuts the terminal error vs open-loop
# ---------------------------------------------------------------------------
def test_mpc_beats_open_loop_under_disturbance():
    from cosmos import cosmos_mpc as e4
    summary, _, _ = e4.run(sigma=0.03, trials=6, verbose=False)
    assert summary["mpc"]["err"] < summary["open"]["err"], \
        f"MPC err {summary['mpc']['err']:.3f} should beat open-loop {summary['open']['err']:.3f}"


# ---------------------------------------------------------------------------
# T15 - E5: the chance-constraint radius inflation calibrates the safety level
# ---------------------------------------------------------------------------
def test_chance_constraint_is_calibrated_and_monotone():
    from cosmos._core import chance_margin
    # k_beta is increasing in beta and zero at 0.5 (median)
    assert abs(chance_margin(1.0, 0.5)) < 1e-6
    assert chance_margin(1.0, 0.99) > chance_margin(1.0, 0.90) > chance_margin(1.0, 0.80) > 0
    from cosmos import cosmos_chance as e5
    rows, _ = e5.run(betas=(0.5, 0.9, 0.99), samples=2000, verbose=False)
    by = {r["beta"]: r for r in rows}
    # higher target beta -> higher realized safe fraction and higher Delta-v
    assert (1 - by[0.99]["rate"]) > (1 - by[0.9]["rate"]) > (1 - by[0.5]["rate"])
    assert by[0.99]["dv"] > by[0.5]["dv"]
    # calibrated: the high-safety level is at least roughly met (conservative is ok)
    assert (1 - by[0.99]["rate"]) >= 0.95


# ---------------------------------------------------------------------------
# T16 - COSMOS matches the SCP/SQP optimum on the constrained problem (near-optimal)
# ---------------------------------------------------------------------------
def test_si_mission_gives_credible_numbers():
    """The realistic SI mission reports sane per-chaser Delta-v (order 0.1-10 m/s)
       and a positive saving from the eccentric metric."""
    from cosmos import cosmos_mission as mis
    r = mis.run(perigee_km=600, apogee_km=8000, N=70, verbose=False)
    per_chaser = r["dvc"] / len(r["starts"])
    assert 0.1 < per_chaser < 10.0, f"per-chaser Δv {per_chaser:.3f} m/s is not credible"
    assert r["dvc"] < r["dvb"], "eccentric metric should save fuel on the eccentric orbit"
    assert r["orbit"]["e"] > 0.3, "scenario should be genuinely eccentric"


def test_eccentric_advantage_is_a_robust_regime():
    """The eccentric-metric saving is positive across the small-to-moderate
       amplitude envelope and grows with eccentricity -- not a cherry-picked point."""
    from cosmos import cosmos_validity as val
    r = val.run(eccentricities=(0.1, 0.3), amplitudes=(0.03, 0.1), N=70, verbose=False)
    grid = r["grid"]
    # every cell in the envelope saves fuel
    assert (grid > 0).all(), f"some cells non-positive:\n{grid}"
    # saving grows with eccentricity (col for e=0.3 > col for e=0.1), each row
    assert (grid[:, 1] > grid[:, 0]).all(), "saving should grow with eccentricity"


def test_numerical_robustness_across_N():
    """The covariant descent stays exact (vs the closed form) and the Cholesky
       preconditioner stays accurate across a wide range of discretizations N,
       despite the metric's condition number growing ~N^2."""
    from cosmos import cosmos_robustness as rob
    for N in (40, 150):
        rel, ratio = rob.descent_accuracy(N, iters=2500)
        assert rel < 1e-6, f"N={N}: descent off the closed form by {rel:.2e}"
        assert 0.95 <= ratio <= 1.05, f"N={N}: LQ ratio {ratio:.3f} out of band"
        assert rob.spd_inverse_accuracy(N) < 1e-8
    # conditioning grows but is finite and monotone
    assert rob.conditioning(150) > rob.conditioning(40)


def test_cosmos_matches_scp_on_constrained_problem():
    from cosmos import cosmos_benchmark as bench
    N = 40; nm = 1.0; T = np.pi; dt = T / (N + 1); w_bc = 20000.0
    start = np.array([3.0, 5.0]); goal = np.array([0.0, 0.0])
    obs = [{"c": np.array([2.5, 2.9]), "r": 1.1}]
    scp, _, ok = bench.solve_scp(start, goal, nm, N, dt, obs, w_bc)
    cos, _ = bench.solve_cosmos(start, goal, nm, N, dt, obs, w_bc)
    # both must respect the keep-out
    assert bench.clearance(scp, start, goal, obs) > -1e-3
    assert bench.clearance(cos, start, goal, obs) > -1e-3
    dv_scp, _ = delta_v(scp, start, goal, dt, nm)
    dv_cos, _ = delta_v(cos, start, goal, dt, nm)
    # COSMOS is within a few percent of the general-purpose constrained optimum
    assert abs(dv_cos - dv_scp) / dv_scp < 0.05, \
        f"COSMOS Δv {dv_cos:.3f} vs SCP {dv_scp:.3f} (gap {100*abs(dv_cos-dv_scp)/dv_scp:.1f}%)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))