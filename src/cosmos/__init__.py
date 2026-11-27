"""COSMOS -- energy-aware formation reconfiguration via covariant optimization
in the Clohessy-Wiltshire metric.

Public API
----------
The package is organized in layers (see the README architecture diagram). The
most-used entry points are re-exported here so consumers do not have to reach
into private submodules:

    from cosmos import chomp_cw, cw_operators, delta_v          # the fuel-aware core
    from cosmos import controllability_gramian, lq_optimal_energy  # the LQ benchmark
    from cosmos import project_metric, project_euclid           # metric-aware safety

The standalone demonstration modules (``cosmos.cosmos_optimal``,
``cosmos.cosmos_safe``, ``cosmos.cosmos_full``, ...) each expose ``run()`` and
``main()`` and are also runnable as scripts (``python -m cosmos.<name>``).
"""

__version__ = "2.1.0"

# --- fuel-aware Clohessy-Wiltshire core (Part III, the contribution) --------
from .chomp_cw import (
    chomp_cw,
    cw_operators,
    cw_offset,
    cw_control,
    delta_v,
    cw_unconstrained_optimum,
    to_flat,
    to_traj,
    linear_relative_operators,
    cw_coeffs,
    ss_coeffs,
)

# --- shared numerical core (LQ benchmark + metric-aware safety) -------------
from ._core import (
    cw_state_matrices,
    controllability_gramian,
    lq_optimal_energy,
    lq_optimal_trajectory,
    project_euclid,
    project_metric,
    project_inter_agent_metric,
    spd_inverse,
    time_varying_operators,
    time_varying_offset,
    tschauner_hempel_coeffs,
    chance_margin,
    inflate_obstacles,
)

__all__ = [
    "__version__",
    "chomp_cw", "cw_operators", "cw_offset", "cw_control", "delta_v",
    "cw_unconstrained_optimum", "to_flat", "to_traj",
    "linear_relative_operators", "cw_coeffs", "ss_coeffs",
    "cw_state_matrices", "controllability_gramian",
    "lq_optimal_energy", "lq_optimal_trajectory",
    "project_euclid", "project_metric", "project_inter_agent_metric", "spd_inverse",
    "time_varying_operators", "time_varying_offset", "tschauner_hempel_coeffs",
    "chance_margin", "inflate_obstacles",
]
