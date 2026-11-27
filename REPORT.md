# COSMOS — Energy-Aware Formation Reconfiguration via Covariant Optimization in the Clohessy–Wiltshire Metric

*Technical report. Every number below is reproduced by a script in this repository; the exact command is given in each caption. Figures live in [`figures/`](figures/).*

---

## Abstract

COSMOS plans fuel-optimal relative trajectories for one or several satellites in low Earth orbit. It takes the covariant gradient of CHOMP — a motion planner from robotics — and makes its smoothness metric aware of orbital mechanics. The single change is to replace the raw-acceleration metric $A^\top A$ with a *control-energy* metric $B^\top B$, where $B = D_2 - G$ encodes the linearized relative dynamics. "Smoothness" then becomes the actual thrust the engine must provide, so the planner spontaneously exploits natural orbital drift. On a four-satellite reconfiguration this saves **40 % ± 10 %** of the realized Δv versus a dynamics-blind planner. The construction is validated against an analytical optimum (it matches the linear-quadratic / Gramian solution to **1.00×**), hardened for safety (a closed-form Riemannian keep-out projection cuts the safety premium from **15× to 1.69×** while staying collision-free), and benchmarked against a general-purpose sequential-quadratic-programming solver (**< 1 %** Δv gap where the baseline converges, **15–18×** faster at larger horizons, and feasible where the baseline is not). The same construction extends to time-varying dynamics around eccentric orbits, where a validity study maps a robust **3–16 %** saving regime and documents where the underlying linearization ceases to hold.

---

## 1. Problem statement

Reconfiguring a satellite formation — or sending chasers to rendezvous with debris — requires firing thrusters, and propellant is the mission's non-renewable budget. The cost currency is Δv (the integrated commanded velocity change). We seek relative trajectories from a start configuration to a goal configuration that **minimize Δv** while (i) respecting the true orbital dynamics, (ii) avoiding keep-out zones, and (iii) keeping agents separated.

The relative motion of satellites near a reference orbit is well approximated by the linear **Clohessy–Wiltshire (CW)** equations in the rotating Hill frame ($x$ radial, $y$ along-track):

$$\ddot x - 2n\dot y - 3n^2 x = u_x, \qquad \ddot y + 2n\dot x = u_y,$$

with mean motion $n$. The terms in $n$ are acceleration the gravity field supplies **for free**; $u=(u_x,u_y)$ is the thrust the engine must actually provide. A 3D cross-track mode $\ddot z + n^2 z = u_z$ is appended when needed.

## 2. Key insight and method

**The metric is the dynamics.** Standard CHOMP minimizes a smoothness functional equal to the energy of the raw acceleration, $\tfrac12\int\|\ddot\xi\|^2 dt$, and descends it with a covariant (metric-preconditioned) gradient. In orbit the raw acceleration is the wrong quantity: part of it is free. COSMOS instead penalizes the **thrust**:

$$F_\text{fuel} = \tfrac12\int\big\lVert \ddot\xi - f_\text{CW}(\xi)\big\rVert^2 dt.$$

Because CW is linear, the discretized thrust is affine in the trajectory, $u = B\,\xi + c_0$ with $B = D_2 - G_\text{CW}$ ($D_2$ the second-difference operator, $G_\text{CW}$ the CW coupling). The smoothness metric becomes $M = B^\top B$, and the covariant descent is

$$\xi \leftarrow \xi - \alpha\, M^{-1}\,\nabla U(\xi), \qquad U = \lambda F_\text{fuel} + F_\text{obs} + \mu F_\text{agents}.$$

As $n\to 0$ the free acceleration vanishes, $B\to D_2$, and COSMOS reduces exactly to free-space CHOMP — a built-in consistency check (test T1). Everything else (obstacle gradient, multi-agent decoupling) is inherited unchanged.

**The preconditioner is an exact Gauss–Newton step.** The obstacle-free cost is exactly quadratic, so its Hessian is $H = B^\top B = M$: the covariant preconditioner is the *exact inverse Hessian*. A single unit step therefore reaches the closed-form optimum from any start (test T11), so the optimality result below is a property of the method, not of tuning.

## 3. Results

### 3.1 Fuel-aware planning (the contribution)
On a four-satellite launch-train → passive 2:1 relative-orbit reconfiguration with a central keep-out, fuel-aware planning spends **29.2** (normalized Δv) versus **49.3** for the dynamics-blind baseline — a **+40.8 %** saving — while remaining collision-free (minimum pair clearance $+1.88$). Over 10 random scenarios the saving is **40 % ± 10 %**.
*Reproduce: `cosmos` · `python -m cosmos.cosmos_validation`. Figure: `figures/cosmos_reconfig.png`.*

### 3.2 Provable optimality (Module A)
For the obstacle-free CW transfer the minimum-energy solution is a linear-quadratic problem with a closed form via the controllability Gramian $W(T)$: $J^\star = \Delta s^\top W(T)^{-1}\Delta s$, with $\Delta s = s_f - \Phi(T)s_0$. Fuel-aware CHOMP recovers it to **1.00×**; the dynamics-blind planner sits at **1.79×**.
*Reproduce: `python -m cosmos.cosmos_optimal`. Figure: `figures/cosmos_optimal.png`.*

### 3.3 Metric-aware safety (Module B)
A metric-blind (Euclidean) keep-out projection injects **15×** the optimal energy. Projecting instead in the fuel metric $M=B^\top B$ — a closed-form Riemannian correction $\delta\xi = M^{-1}a\,(b-a^\top\xi)/(a^\top M^{-1}a)$ — cuts the premium to **1.69×** while staying strictly collision-free (89 % of the wasted energy recovered).
*Reproduce: `python -m cosmos.cosmos_safe`. Figure: `figures/cosmos_safe.png`.*

### 3.4 Benchmark against a serious baseline
On the hard (non-convex keep-out) problem, the analytical LQ optimum is **infeasible** (it has no obstacle notion). Against a general-purpose SQP solver (SciPy SLSQP minimizing the exact control energy subject to the hard keep-out), COSMOS matches the optimum to **< 1 % Δv** where SLSQP converges ($N\le 60$), and is **15–18× faster** at $N=100\text{–}150$ — where SLSQP exhausts its iteration budget and returns an infeasible trajectory, while COSMOS stays feasible. SLSQP factorizes dense Jacobians that grow with the horizon (≈ cubic per iteration); the covariant step is a fixed structured solve.
*Reproduce: `python -m cosmos.cosmos_benchmark`. Figure: `figures/cosmos_benchmark.png`.*

### 3.5 Generality to eccentric orbits
Around an eccentric reference the relative dynamics are time-varying (Tschauner–Hempel): $B = D_2 - G(t)$. The same construction applies with only the block $G$ changing. Scored against the full nonlinear relative dynamics, the Δv saving of the correct time-varying metric over the circular (CW) metric rises from ~0 % at $e=0$ (validating CW for circular LEO) to **~16 % at $e=0.3$**, holding in 2D and 3D.
*Reproduce: `python -m cosmos.cosmos_eccentric`. Figure: `figures/cosmos_eccentric.png`.*

### 3.6 Extensions
- **Inter-agent safety (E3):** the Riemannian projection, extended to the coupled pairwise constraint $\lVert x_i(t)-x_j(t)\rVert \ge r$, keeps a swarm collision-free at **~1.3×** lower Δv than a metric-blind push.
- **Receding-horizon control (E4):** wrapping the solver in an MPC loop cuts the terminal error **~10×** under process noise; each re-plan is essentially one Gauss–Newton step.
- **Chance constraints (E5):** inflating the keep-out radius by $\Phi^{-1}(\beta)\sigma$ turns it into a calibrated chance constraint using the *same* projection; Monte Carlo confirms the realized safety tracks the target $\beta$.
*Reproduce: `python -m cosmos.cosmos_swarm_safe` · `cosmos_mpc` · `cosmos_chance`.*

## 4. Validity and honest limits

The eccentric-metric advantage is a **regime**, not a single point: a sweep over (eccentricity × transfer amplitude), scored under the full nonlinear dynamics, shows a positive saving growing with eccentricity across small-to-moderate amplitudes (**3–16 %**). The advantage tracks the dynamics mismatch (how wrong the circular assumption is).

It is a *linear* method, and we state its limits plainly:
- The core optimizes **control energy (L2)**, the smooth quantity that makes the metric, its preconditioner and the closed-form optimum possible; we *report* the realized **Δv (L1)**, which matches the continuous low-thrust (electric-propulsion) regime of slow reconfiguration.
- The **1.00×** optimality holds only for the obstacle-free (convex) problem; with obstacles or coupled agents there is no closed form — which is exactly where the iterative planner is needed.
- Tschauner–Hempel is a **linearization**: for very large-amplitude or strongly-radial transfers the linear model itself departs from the true nonlinear motion, so no linear metric (CW or TH) is the right object there. The validity map makes this envelope explicit rather than hiding it.
- Obstacles are modeled as spheres; keep-out constraints are linearized per waypoint.
*Reproduce: `python -m cosmos.cosmos_validity`. Figure: `figures/cosmos_validity.png`.*

## 5. Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python run_all.py            # mission + every demo/analysis + figures + tests
python -m pytest -q          # 18 regression tests lock every headline claim
```

The regression suite (T1–T16, plus the 3D-eccentric and validity tests) encodes each claim above as an automatic check, so a refactor cannot silently break a result.

## 6. References

1. N. Ratliff, M. Zucker, J. A. Bagnell, S. Srinivasa. *CHOMP: Gradient optimization techniques for efficient motion planning.* IEEE ICRA, 2009.
2. M. Zucker et al. *CHOMP: Covariant Hamiltonian optimization for motion planning.* IJRR, 2013.
3. W. H. Clohessy, R. S. Wiltshire. *Terminal guidance system for satellite rendezvous.* J. Aerospace Sci., 1960.
4. J. Tschauner, P. Hempel. *Rendezvous with a target in an elliptical orbit.* Astronautica Acta, 1965.
5. S. A. Schweighart, R. J. Sedwick. *High-fidelity linearized J2 model for satellite formation flight.* JGCD, 2002.
6. A. E. Bryson, Y.-C. Ho. *Applied Optimal Control* (LQ regulator, controllability Gramian). Hemisphere, 1975.
7. J. Nocedal, S. J. Wright. *Numerical Optimization* (Gauss–Newton, SQP). Springer, 2006.
8. X. Liu, P. Lu, B. Pan. *Survey of convex optimization for aerospace applications* (sequential convex programming). Astrodynamics, 2017.
9. H. W. Kuhn. *The Hungarian method for the assignment problem.* Naval Research Logistics Q., 1955.
10. H. Schaub, J. L. Junkins. *Analytical Mechanics of Space Systems.* AIAA, 2018.

---

<sub>MIT License. This report summarizes the method and results; the code is the executable reference — each module's header documents what it implements, and `run_all.py` regenerates every figure and number above.</sub>
