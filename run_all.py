#!/usr/bin/env python3
"""
COSMOS — run the WHOLE framework in one command
================================================

Runs the mission (program + results), every demonstration (figures), the analyses,
and the test suite, end to end. All figures are written to ``figures/``; all numeric
results are printed to the console.

Usage
-----
    python run_all.py                # everything, incl. multi-seed validation (a few min)
    python run_all.py --fast         # skip the slow multi-seed sweeps (~1-2 min total)
    python run_all.py --smoke        # just a few quick steps, to check the harness
    python run_all.py --no-tests     # don't run pytest at the end

The real-time pygame viewer is interactive and is intentionally NOT part of the batch.
Run it on its own:  python -m cosmos.pygame_orbital_sim   (needs the ".[viz]" extra + a display)
The browser widget is also standalone:  open cosmos_demo.html
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def run(title, module_args, cwd=None):
    cwd = cwd or FIG
    print(f"\n{'='*78}\n▶ {title}\n{'='*78}")
    t0 = time.time()
    r = subprocess.run([sys.executable, *module_args], cwd=str(cwd))
    status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"  ↳ {status} in {time.time()-t0:.1f}s")
    return r.returncode == 0


# (title, args)  — args passed to the current Python interpreter
CORE = [
    ("Mission · reconfiguration (fuel · riemann · hungarian · 2D)",
     ["-m", "cosmos.cli", "--out", "cosmos_mission_reconfig.png"]),
    ("Mission · debris rendezvous (3D · SI units)",
     ["-m", "cosmos.cli", "--scenario", "debris", "--dim", "3", "--units", "si",
      "--out", "cosmos_mission_debris3d.png"]),
    ("Mission · dynamics-blind baseline (ablation, results only)",
     ["-m", "cosmos.cli", "--metric", "free", "--safety", "none", "--assign", "fixed",
      "--mode", "results"]),
    ("Contribution · fuel-aware Clohessy–Wiltshire metric",
     ["-m", "cosmos.chomp_cw"]),
    ("Module A · analytical LQ optimum (1.00×) and the 15× pitfall",
     ["-m", "cosmos.cosmos_optimal"]),
    ("Module B · Riemannian safe projection (15× → 1.69×)",
     ["-m", "cosmos.cosmos_safe"]),
    ("Realistic LEO units (m/s)",
     ["-m", "cosmos.cosmos_units"]),
    ("Full system · multi-agent rendezvous (2D)",
     ["-m", "cosmos.cosmos_full"]),
    ("Full system · multi-agent rendezvous (3D)",
     ["-m", "cosmos.cosmos_full", "--3d"]),
    ("Generality · B = D₂ − G across free / CW / J2 (two coefficients)",
     ["-m", "cosmos.cosmos_generality"]),
    ("Motivation · CW-passive orbit vs realistic J2 (Schweighart–Sedwick)",
     ["-m", "cosmos.cw_vs_j2_demo"]),
    ("Swarm · price of decoupling (joint vs Gauss–Seidel)",
     ["-m", "cosmos.cosmos_joint"]),
    ("Pedagogy · single-agent covariant CHOMP",
     ["-m", "cosmos.chomp_single_agent"]),
    ("Pedagogy · multi-agent decoupling strategies",
     ["-m", "cosmos.chomp_swarm"]),
]

SLOW = [
    ("Validation · multi-seed robustness (53% ± …, 100% safe)",
     ["-m", "cosmos.cosmos_validation"]),
    ("Validation · sweep with 95% confidence intervals",
     ["-m", "cosmos.cosmos_sweep"]),
]

SMOKE = CORE[:1] + CORE[9:12]   # one mission + the three new analyses


def main():
    p = argparse.ArgumentParser(description="Run the whole COSMOS framework.")
    p.add_argument("--fast", action="store_true",
                   help="skip multi-seed validation; run the sweep in --quick mode")
    p.add_argument("--smoke", action="store_true",
                   help="run only a few quick steps (harness check)")
    p.add_argument("--no-tests", action="store_true", help="skip pytest")
    args = p.parse_args()

    FIG.mkdir(exist_ok=True)
    t0 = time.time()
    before = {f.name for f in FIG.glob("*.png")}

    steps = list(SMOKE) if args.smoke else list(CORE)
    if not args.smoke:
        if args.fast:
            steps.append(("Validation · sweep with 95% CIs (quick)",
                          ["-m", "cosmos.cosmos_sweep", "--quick"]))
        else:
            steps += SLOW

    results = [(title, run(title, a)) for title, a in steps]

    if not args.no_tests and not args.smoke:
        results.append(("Test suite (pytest)", run("Test suite (pytest)",
                                                   ["-m", "pytest", "-q"], cwd=ROOT)))

    # ---- summary ----
    after = sorted(FIG.glob("*.png"), key=lambda f: f.stat().st_mtime)
    new = [f.name for f in after if f.name not in before]
    print(f"\n{'#'*78}\n# COSMOS — run complete in {time.time()-t0:.0f}s\n{'#'*78}")
    ok = sum(1 for _, s in results if s)
    print(f"steps: {ok}/{len(results)} ok")
    for title, s in results:
        print(f"   {'✓' if s else '✗'}  {title}")
    print(f"\nfigures in {FIG}/ ({'updated: ' + ', '.join(new) if new else 'no new files'})")
    print("interactive extras (not in batch):  python -m cosmos.pygame_orbital_sim   |   "
          "open cosmos_demo.html")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
