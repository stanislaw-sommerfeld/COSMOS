"""
COSMOS - The preconditioner is the exact Gauss-Newton step (second-order, not tuned)
====================================================================================

A fair question about any CHOMP-style result: is the "1.00x optimal" number a
property of the method, or an artifact of a well-chosen step size and iteration
budget? This module answers it cleanly.

The fuel cost is exactly quadratic:  U(xi) = 1/2 ||B xi + c0||^2  (no obstacle).
Its gradient is  g = B^T (B xi + c0)  and its Hessian is  H = B^T B = M. The
covariant preconditioner COSMOS uses is  M^-1 -- i.e. the EXACT inverse Hessian.
So one covariant step with unit length is a full Newton (here Gauss-Newton,
exact because the model is quadratic) step and lands on the global optimum:

    xi* = xi0 - M^-1 g(xi0)        (exact, for any xi0, in ONE step)

Consequences, all checked numerically below:
  (1) From ANY start, a single unit step reaches the closed-form optimum to
      machine precision -- the result does not depend on the iteration count.
  (2) The gradient-descent residual contracts linearly with a rate set by the
      step size; the preconditioned (Newton) step is the alpha=1 limit.
  (3) The optimum is start-independent (convex quadratic), so warm-starting
      changes cost-per-step, not the answer.

This reframes the optimizer from "a path smoother that happens to converge" to
"a second-order solver for the LQ transfer, with CHOMP's covariant structure as
the exact Gauss-Newton preconditioner" -- a stronger, defensible claim.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .chomp_cw import (cw_operators, cw_offset, to_flat, to_traj,
                      cw_unconstrained_optimum)


def _quadratic_pieces(start, goal, nm, n, dt, w_bc):
    B, Minv, blocks = cw_operators(n, dt, nm, w_bc=w_bc, dim=len(start))
    c0 = cw_offset(blocks, start, goal)
    return B, Minv, c0


def one_step_error(start, goal, nm, n, dt, w_bc=0.0, xi0=None):
    """Relative distance from (xi0 - M^-1 g(xi0)) to the closed-form optimum.
       Default xi0 = straight line; pass any xi0 to show start-independence."""
    B, Minv, c0 = _quadratic_pieces(start, goal, nm, n, dt, w_bc)
    if xi0 is None:
        xi0 = to_flat(np.linspace(start, goal, n + 2)[1:-1])
    g = B.T @ (B @ xi0 + c0)
    xi1 = xi0 - Minv @ g                       # one unit covariant (Newton) step
    cf = to_flat(cw_unconstrained_optimum(start, goal, nm=nm, n=n, dt=dt, w_bc=w_bc))
    return np.linalg.norm(xi1 - cf) / np.linalg.norm(cf)


def descent_residuals(start, goal, nm, n, dt, step, iters, w_bc=0.0):
    """Residual ||xi_k - xi*|| per iteration for a fixed-step covariant descent."""
    B, Minv, c0 = _quadratic_pieces(start, goal, nm, n, dt, w_bc)
    cf = to_flat(cw_unconstrained_optimum(start, goal, nm=nm, n=n, dt=dt, w_bc=w_bc))
    xi = to_flat(np.linspace(start, goal, n + 2)[1:-1])
    res = []
    for _ in range(iters):
        g = B.T @ (B @ xi + c0)
        xi = xi - step * (Minv @ g)
        res.append(np.linalg.norm(xi - cf) / np.linalg.norm(cf))
    return np.array(res)


def run(nm=1.0, N=60, T=np.pi, verbose=True):
    dt = T / (N + 1)
    start = np.array([3.0, 5.0]); goal = np.array([0.0, 0.0])
    rng = np.random.default_rng(0)

    # (1) one unit step from the straight line
    e_line = one_step_error(start, goal, nm, N, dt)
    # (1b) one unit step from three random starts -> same optimum
    e_rand = [one_step_error(start, goal, nm, N, dt,
                             xi0=to_flat(np.linspace(start, goal, N + 2)[1:-1])
                             + rng.normal(0, 2.0, 2 * N))
              for _ in range(3)]
    # (2) residual curves for a few step sizes (alpha=1 is the Newton step)
    steps = [0.1, 0.3, 0.6, 1.0]
    curves = {s: descent_residuals(start, goal, nm, N, dt, s, 60) for s in steps}

    if verbose:
        print("=== COSMOS: the covariant preconditioner is the exact Gauss-Newton step ===")
        print(f"one unit step from the straight line   -> rel. error {e_line:.2e}")
        print(f"one unit step from 3 random starts     -> rel. error "
              f"{', '.join(f'{v:.1e}' for v in e_rand)}")
        print("  => a SINGLE unit covariant step solves the unconstrained problem to")
        print("     machine precision, from any start. The '1.00x optimal' result is a")
        print("     property of the method (M = B^T B is the exact Hessian), not of the")
        print("     iteration budget or step-size tuning.")
        for s in steps:
            print(f"  step alpha={s:3.1f}: residual after 60 iters = {curves[s][-1]:.2e}"
                  + ("   <- Newton step (exact)" if s == 1.0 else ""))
    return dict(e_line=e_line, e_rand=e_rand, steps=steps, curves=curves)


def main(out="cosmos_newton.png"):
    r = run(verbose=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(r["steps"])))
    for col, s in zip(cmap, r["steps"]):
        c = r["curves"][s]
        lbl = f"α = {s}" + ("  (exact Newton)" if s == 1.0 else "")
        ax.semilogy(np.arange(1, len(c) + 1), np.maximum(c, 1e-16),
                    lw=2.4 if s == 1.0 else 1.8, color=col, label=lbl)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"relative error $\|\xi_k-\xi^*\|/\|\xi^*\|$")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    ax.set_title("Covariant descent in the fuel metric: α=1 is a one-step exact solve\n"
                 "(the preconditioner $M^{-1}=(B^\\top B)^{-1}$ is the exact Gauss-Newton Hessian inverse)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
