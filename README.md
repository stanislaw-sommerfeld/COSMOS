<div align="center">

# 🛰️ COSMOS

### Energy-Aware Formation Reconfiguration via Covariant Optimization in the Clohessy–Wiltshire Metric

*Covariant Optimization for Swarm Maneuvers in Orbital Space*

[![Python](https://img.shields.io/badge/Cosmos-Website-blue)](https://stanislaw-sommerfeld.github.io/COSMOS/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22D3EE.svg)](LICENSE)

<br/>

<img src="figures/cosmos_reconfig.png" alt="Formation reconfiguration: launch train to passive 2:1 relative orbit" width="640"/>

*A four-satellite launch train reconfigures into a passive 2:1 relative orbit. Solid curves (fuel-aware) ride the natural orbital drift; dashed curves (free-space) push straight through it.*

</div>

---

## ✨ What is COSMOS?

COSMOS plans **fuel-optimal trajectories for a swarm of satellites** moving relative to one another in low Earth orbit. It takes a motion-planning idea — the **covariant gradient descent of [CHOMP](https://ieeexplore.ieee.org/document/5152817)** — and makes it *aware of orbital mechanics*.

The one-sentence idea:

> **Standard trajectory smoothing minimizes acceleration. In orbit, the dynamics already supply free acceleration — so minimizing raw acceleration optimizes the wrong thing. COSMOS replaces the smoothness metric with one built from the Clohessy–Wiltshire dynamics, so "smoothness" becomes the actual thrust the engine must provide.**

That single substitution — from $A^\top A$ (raw acceleration) to $B^\top B$ (thrust energy in the orbital frame) — turns a generic path smoother into a fuel-aware orbital maneuver planner, and it does so while keeping CHOMP's elegant covariant structure intact.

## 🎯 Key results

| Metric | Result |
|---|---|
| Δv saving vs. dynamics-blind planning | **40% ± 10%** (validated over 10 random scenarios) |
| Optimality vs. analytical LQ optimum | **1.00×** (provably optimal in the energy sense) |
| Safe keep-out avoidance cost | **15× → 1.69×** (89% of wasted energy recovered) |
| Collision-free success rate | **100%** (hardened mode) |
| Realistic LEO (500 km) per-satellite cost | **~1–2 m/s**, arriving at cm/s |

**Extensions (v2)** — the same construction, pushed further:

| Extension | Result |
|---|---|
| Time-varying metric for **eccentric** orbits (TH) | **~16%** Δv saved at $e=0.3$ (0% at $e=0$, validating CW for circular LEO) |
| Exact **Gauss–Newton** optimality | the 1.00× optimum is reached in a **single** covariant step from any start |
| Metric-aware **inter-agent** safety | collision-free at **~1.3× lower Δv** than the metric-blind push |
| **Receding-horizon** (MPC) | terminal error cut **~10×** under process noise |
| **Chance-constrained** keep-outs | realized safety calibrated to the target $\beta$ under navigation uncertainty |
| **Benchmark** vs SCP/SQP (SLSQP) | within **<1%** of the constrained optimum, **15–18× faster** at $N\!=\!100\text{–}150$, and feasible where SLSQP is not |

## 💡 The core idea

CHOMP minimizes a functional combining smoothness and obstacle cost, and descends it with a **covariant** (metric-preconditioned) gradient:

$$U(\xi) = \lambda\, F_\text{fuel}(\xi) + F_\text{obs}(\xi) + \mu\, F_\text{agents}(\xi), \qquad \xi \leftarrow \xi - \alpha\, (B^\top B)^{-1}\, \nabla U(\xi)$$

The fuel term is the energy of the **thrust** — the part of the acceleration the dynamics do *not* provide for free:

$$F_\text{fuel} = \tfrac{1}{2}\!\int \big\lVert \ddot{\xi} - f_\text{CW}(\xi)\big\rVert^2\, dt, \qquad B = D_2 - G_\text{CW}$$

where $D_2$ is the discrete second-derivative operator and $G_\text{CW}$ encodes the linearized Clohessy–Wiltshire dynamics (the in-plane radial/along-track coupling, plus a decoupled cross-track oscillator in 3D). When the orbital rate $n \to 0$, $B \to D_2$ and COSMOS gracefully reduces to ordinary free-space CHOMP — a built-in sanity check the code verifies.

> **A note on honesty:** we optimize **control energy (L2)**, the smooth quantity that makes the Riemannian metric, its preconditioner, and the closed-form optimal benchmark possible. We *report* the realized **Δv (L1)**, the true propellant currency. This matches the continuous, low-thrust (electric-propulsion) regime that slow formation reconfiguration actually uses.

## ⚙️ Installation

```bash
git clone https://github.com/stanislaw-sommerfeld/cosmos.git
cd cosmos
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .                                   # add ".[viz]" for the pygame viewer
```

This installs the `cosmos` package (src-layout) and a `cosmos` command. Prefer plain dependencies instead? `pip install -r requirements.txt`.

## ▶️ Run the whole framework at once

One command runs the mission, every demonstration and analysis, regenerates **all figures** into `figures/`, prints **all numeric results**, and runs the **test suite**:

```bash
python run_all.py              # everything, incl. multi-seed validation — a few minutes
python run_all.py --fast       # skip the slow multi-seed sweeps (~1–2 min)
python run_all.py --smoke      # just a few quick steps (harness check)
python run_all.py --no-tests   # don't run pytest at the end
```

…or, equivalently, with `make`:

```bash
make all      # = python run_all.py        (full run + tests)
make fast     # = python run_all.py --fast
make test     # just the test suite
```

Figures are written to `figures/`; results print to the console. The two **interactive** pieces are standalone and intentionally *not* part of the batch:

```bash
python -m cosmos.pygame_orbital_sim   # real-time viewer (needs ".[viz]" + a display)
open cosmos_demo.html                 # interactive covariant-descent widget (any browser)
```

## 🚀 Usage

**The unified driver** — one command, configured by flags:

```bash
cosmos                                              # reconfig · fuel · riemannian · hungarian · 2D
cosmos --metric free --safety none --assign fixed   # dynamics-blind baseline (ablation)
cosmos --scenario debris --dim 3 --units si         # 3D debris rendezvous, real units
cosmos --units si --mode sim --out figure.png       # render a figure only
```

(`cosmos ...` is equivalent to `python -m cosmos.cli ...`.)

| Flag | Options |
|---|---|
| `--scenario` | `reconfig` · `debris` · `swarm` · `single` |
| `--metric` | `free` · `fuel` |
| `--safety` | `none` · `euclid` · `riemann` |
| `--assign` | `fixed` · `hungarian` |
| `--dim` | `2` · `3` |
| `--units` | `norm` · `si` |
| `--mode` | `sim` · `results` · `both` |

**Reproduce the headline figures** — each demonstration script is a package module, run with `python -m`:

```bash
python -m cosmos.chomp_cw          # the fuel-aware contribution
python -m cosmos.cosmos_validation # robustness (slowest; a few minutes)
python -m cosmos.cosmos_units      # credible m/s numbers
python -m cosmos.cosmos_optimal    # 1.00× LQ optimum + the 15× pitfall   (Module A)
python -m cosmos.cosmos_safe       # 15× → 1.69× safety fix                (Module B)
python -m cosmos.cosmos_eccentric  # time-varying metric, eccentric orbits (extension)
python -m cosmos.cosmos_newton     # exact Gauss–Newton preconditioner     (extension)
python -m cosmos.pygame_orbital_sim   # real-time viewer (needs ".[viz]" + a display)
```

Figures are written to the current directory.

## 🏗️ Architecture

The science is built up in layers, each a self-contained module inside the `cosmos` package. A single driver, [`cosmos/cli.py`](src/cosmos/cli.py), composes them into the full mission and is the one command you call.

```
                       cosmos.cli   ← unified mission driver (the `cosmos` command)
                            │
        ┌───────────────────┼───────────────────────────┐
   fuel-aware CW      optimal benchmark            safe projection
   metric (B^T B)     (LQ / Gramian, Module A)     (Riemannian, Module B)
        │                   │                            │
        └───────────────────┴───────────────────────────┘
                            │
              covariant CHOMP core  +  multi-agent decoupling
              (chomp_single_agent)     (chomp_swarm: Jacobi / Gauss–Seidel / priority)
```

| Stage | Module (`python -m cosmos.<name>`) | What it demonstrates |
|---|---|---|
| 1 | `chomp_single_agent` | Covariant CHOMP — one agent, one obstacle |
| 2 | `chomp_swarm` | Multi-agent planning, three decoupling strategies |
| 3 | `orbital_sim` · `pygame_orbital_sim` | Animation (GIF) and real-time viewer |
| 4 | `chomp_cw` | **The contribution:** fuel-aware CW metric $B^\top B$ |
| 5 | `cosmos_full` `[--3d]` | Full system: multi-agent + CW + rendezvous |
| 6 | `cosmos_validation` | Multi-seed robustness (40% ± 10%, 100% safe) |
| 7 | `cosmos_units` | Realistic LEO SI units (m/s) |
| 8 | `cosmos_optimal` | **Module A:** analytical LQ optimum (1.00×) |
| 9 | `cosmos_safe` | **Module B:** Riemannian safe projection (15× → 1.69×) |
| 10 | `cosmos_eccentric` | **Extension E1:** time-varying metric for eccentric orbits (TH) |
| 11 | `cosmos_newton` | **Extension E2:** the preconditioner is the exact Gauss–Newton step |
| 12 | `cosmos_swarm_safe` | **Extension E3:** metric-aware inter-agent safety (coupled Riemannian) |
| 13 | `cosmos_mpc` | **Extension E4:** receding-horizon replanning under disturbance |
| 14 | `cosmos_chance` | **Extension E5:** chance-constrained keep-outs (nav. uncertainty) |
| — | `_core` | Shared numerics: Gramian, LQ optimum, projections, time-varying ops, chance |
| — | `cli` | **Unified driver** composing the above into the mission |

## 📂 Repository structure

```
cosmos/
├── README.md              ← you are here
├── LICENSE                ← MIT
├── run_all.py             ← run the whole framework in one command
├── Makefile               ← make all / fast / test / clean
├── cosmos_demo.html       ← interactive covariant-descent widget (no build)
├── pyproject.toml         ← package metadata, dependencies, `cosmos` entry point
├── requirements.txt       ← plain dependency list (alternative to pip install -e .)
├── .github/workflows/
│   └── ci.yml               runs the test suite on every push / PR
├── tests/
│   └── test_cosmos.py       locks the headline invariants (n→0, 1.00×, collision-free, …)
├── figures/               ← curated result figures + animation
└── src/
    └── cosmos/            ← the importable package
        ├── __init__.py             curated public API (re-exports the core)
        ├── _core.py                shared numerics: Gramian, LQ optimum, projections,
        │                             CW state system, time-varying operators
        ├── cli.py                  unified mission driver
        ├── chomp_single_agent.py   covariant CHOMP core
        ├── chomp_swarm.py          multi-agent decoupling
        ├── chomp_cw.py             fuel-aware CW metric  +  B = D₂ − G generality
        ├── cosmos_full.py          full system (2D / --3d)
        ├── cosmos_generality.py    one solver, any linear dynamics (free / CW / J2)
        ├── cosmos_eccentric.py     E1 — time-varying metric, eccentric orbits
        ├── cosmos_newton.py        E2 — exact Gauss–Newton preconditioner
        ├── cosmos_swarm_safe.py    E3 — metric-aware inter-agent safety
        ├── cosmos_mpc.py           E4 — receding-horizon replanning (MPC)
        ├── cosmos_chance.py        E5 — chance-constrained keep-outs
        ├── cw_vs_j2_demo.py        motivation: a CW-passive orbit is not J2-passive
        ├── cosmos_joint.py         swarm: price-of-decoupling analysis
        ├── cosmos_validation.py    multi-seed validation
        ├── cosmos_sweep.py         validation sweep with 95% confidence intervals
        ├── cosmos_units.py         realistic SI units
        ├── cosmos_optimal.py       Module A — LQ / Gramian benchmark
        ├── cosmos_safe.py          Module B — Riemannian projection
        ├── orbital_sim.py          GIF animation
        └── pygame_orbital_sim.py   real-time viewer
```

## 🔬 The three contributions

**1 · Fuel-aware Clohessy–Wiltshire metric.** Replacing $A^\top A$ with $B^\top B = (D_2 - G_\text{CW})^\top (D_2 - G_\text{CW})$ makes the covariant gradient minimize true thrust energy. The planner spontaneously discovers natural-motion coast arcs.

<div align="center"><img src="figures/chomp_part3_cw.png" width="620"/></div>

**2 · Provable optimality (Module A).** A linear-quadratic formulation of the CW transfer admits a closed-form optimum via the controllability Gramian, giving an *analytical ground truth*. Fuel-aware CHOMP recovers it to **1.00×**, while a dynamics-blind planner sits at 1.79× — and a naive Euclidean keep-out projection balloons to **15×**.

<div align="center"><img src="figures/cosmos_optimal.png" width="620"/></div>

**3 · Metric-aware safety (Module B).** That 15× pitfall is fixed by projecting onto safety constraints *in the fuel metric* $B^\top B$ rather than in Euclidean space — a closed-form Riemannian projection that cuts the safety premium to **1.69×** while staying strictly collision-free.

<div align="center"><img src="figures/cosmos_safe.png" width="620"/></div>

## 🧪 Extensions (v2)

Two additions that answer the two hardest questions about the method.

**4 · A time-varying metric for eccentric orbits (Tschauner–Hempel).** Clohessy–Wiltshire assumes a *circular* reference orbit. Around an **eccentric** orbit the linearized relative dynamics are time-varying, and a CW-"passive" relative orbit drifts by many times its own amplitude per revolution. COSMOS's construction generalizes with a single change — the dynamics block becomes $G(t)$, so $B = D_2 - G(t)$ — and nothing else moves. Planning in the correct time-varying metric and scoring against the **full nonlinear** relative equations of motion, the Δv saving climbs from ~0% at $e=0$ (CW is validated for circular LEO) to **~16% at $e=0.3$**. The metric matters exactly when the dynamics do.

```bash
python -m cosmos.cosmos_eccentric   # saving-vs-eccentricity curve
```

<div align="center"><img src="figures/cosmos_eccentric.png" width="720"/></div>

**5 · The preconditioner is the exact Gauss–Newton step.** The fuel cost is exactly quadratic, so its Hessian is $M = B^\top B$ — precisely the covariant preconditioner. One unit covariant step is therefore a full Newton step: from *any* start it reaches the closed-form optimum to machine precision in a **single iteration**. The "1.00× optimal" headline is a property of the method (the metric *is* the exact inverse Hessian), not of the step size or iteration budget.

```bash
python -m cosmos.cosmos_newton      # single-step convergence
```

**6 · Metric-aware inter-agent safety (E3).** Module B made *static* keep-out avoidance fuel-cheap; a swarm's real hazard is the *other agents*. E3 extends the closed-form Riemannian projection to the **coupled** pairwise constraint $\lVert x_i(t)-x_j(t)\rVert \ge r_{\text{col}}$: the correction is shared between the two agents, each moving the way that is cheapest in its own fuel metric. The swarm stays strictly collision-free at **~1.3× lower Δv** than the metric-blind push — the swarm analogue of Module B.

**7 · Receding-horizon replanning / MPC (E4).** Every other result is open-loop. Wrapping the fuel-aware solver in an MPC loop (execute one step, re-plan from the measured state), closed-loop control cuts the terminal error **~10×** under process noise for a modest Δv premium. Each re-plan is essentially one Gauss–Newton step (contribution 5), so the metric is MPC-ready.

**8 · Chance-constrained keep-outs (E5).** The keep-out radius is treated as exact. Under Gaussian navigation uncertainty $\sigma$, inflating the radius by $\Phi^{-1}(\beta)\,\sigma$ turns it into a chance constraint — "stay outside with probability $\ge \beta$" — enforced by the **same** Module B projection with zero new machinery. Monte Carlo confirms the realized safety tracks the target $\beta$ and Δv grows smoothly with it: a tunable safety/fuel trade.

```bash
python -m cosmos.cosmos_swarm_safe  # E3 · inter-agent metric safety
python -m cosmos.cosmos_mpc         # E4 · receding-horizon replanning
python -m cosmos.cosmos_chance      # E5 · chance-constrained keep-outs
```

**9 · Benchmark against serious baselines (why it is non-trivial *and* fast).** The base problem is linear-quadratic, so a fair critic asks why a planner is needed when the LQ optimum is closed-form (E2). Answer: the LQ form cannot represent a keep-out. Compared on the *same* hard (non-convex keep-out) problem against the analytic LQ optimum (infeasible — no obstacle notion) and a general-purpose SQP solver (scipy SLSQP minimizing the exact control energy subject to the hard keep-out), COSMOS matches the SCP optimum to **<1% Δv** where SLSQP converges, stays **strictly collision-free**, and is **15–18× faster** at $N=100\text{–}150$ — where SLSQP exhausts its iteration budget and returns an infeasible trajectory. SLSQP factorizes dense Jacobians that grow with the horizon; COSMOS's covariant step is a fixed structured solve.

```bash
python -m cosmos.cosmos_benchmark   # COSMOS vs SCP/SQP vs LQ (quality, feasibility, scaling)
```

<div align="center"><img src="figures/cosmos_benchmark.png" width="760"/></div>

> **Code-quality note (v2).** The shared numerics — the controllability Gramian, the CW state system, the closed-form LQ optimum, the time-varying operators, and the Euclidean/Riemannian keep-out and inter-agent projections — now live once in [`cosmos/_core.py`](src/cosmos/_core.py) instead of being copy-pasted across modules. Modules A and B are importable (no side effects at import) and unit-tested. A faster, exact Van Loan Gramian is available via `controllability_gramian(..., method="vanloan")`. The regression suite grew from 8 to **15 tests**; every headline number is unchanged.

## 🖼️ Gallery

| Multi-seed validation | Realistic LEO units | Live animation |
|:--:|:--:|:--:|
| <img src="figures/cosmos_validation.png" width="260"/> | <img src="figures/cosmos_units.png" width="200"/> | <img src="figures/orbital_sim.gif" width="200"/> |

## 📖 Documentation

The full method — the layered derivation, the cost-functional design, and the design decisions — is presented inline above (**The core idea** and **The three contributions**) and in the accompanying project report. The code is the executable reference: each module's header documents what it implements, and `python run_all.py` reproduces every result and figure.

## 📜 License

Released under the [MIT License](LICENSE).

<div align="center">
<br/>
<sub>Built with covariant gradients, a little orbital mechanics, and a healthy respect for delta-v.</sub>
</div>
