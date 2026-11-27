"""
COSMOS - Headline "concrete gain" figures (pedagogy-first)
==========================================================

The result figures elsewhere are correct but the *gain* lives in their captions.
These are the two figures a first-time reader should see, with the gain shown IN
the image:

  * hero (2D): the formation reconfiguration, "without COSMOS" (dashed) vs
    "with COSMOS" (solid), a large +XX% callout, and a per-satellite fuel bar so
    the saving is legible at a glance.
  * hero 3D: the same idea with an out-of-plane (cross-track) component and a
    keep-out sphere, free-space vs fuel-aware, with the Delta-v gain on the plot.
    Also rendered as a short rotating GIF (make_gif).

They reuse the validated mission driver (cli.scenario / assign / solve / metrics)
unchanged, so the numbers match the rest of the project.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import cli

FUEL = "#1d9e75"; BLIND = "#c2502f"; INK = "#14181f"; MUT = "#555c66"
KEEP = "#d96459"


def _solve_both(scenario="reconfig", dim=2, N=60):
    nm = 1.0; T = np.pi; dt = T / (N + 1)
    starts, goals0, obstacles, title = cli.scenario(scenario, dim)
    goals, _ = cli.assign(starts, goals0, "hungarian", nm, T, dim)
    common = dict(nm=nm, n=N, dt=dt, dim=dim, obstacles=obstacles,
                  safety="riemann", iters=900)
    trajs = cli.solve(starts, goals, **common)
    trajs_free = cli.solve(starts, goals, **{**common, "nm": 0.0})
    dv, en, per = cli.total_dv(trajs, starts, goals, dt, nm)
    dvf, enf, perf = cli.total_dv(trajs_free, starts, goals, dt, nm)
    clr = cli.min_pair_clearance(trajs, starts, goals)
    return dict(starts=starts, goals=goals, obstacles=obstacles, trajs=trajs,
                trajs_free=trajs_free, dv=dv, dvf=dvf, per=per, perf=perf,
                clr=clr, saving=100 * (dvf - dv) / dvf, dim=dim)


# ---------------------------------------------------------------------------
# HERO 2D : trajectories + big gain callout + per-satellite fuel bars
# ---------------------------------------------------------------------------
def hero_2d(out="cosmos_hero.png", verbose=True):
    r = _solve_both("reconfig", dim=2)
    K = len(r["starts"])
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, K))

    fig = plt.figure(figsize=(12.6, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1])
    axT = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    th = np.linspace(0, 2 * np.pi, 160)
    for ob in r["obstacles"]:
        axT.fill(ob["c"][1] + ob["r"] * np.cos(th), ob["c"][0] + ob["r"] * np.sin(th),
                 color=KEEP, alpha=0.18)
        axT.text(ob["c"][1], ob["c"][0], "zone\ninterdite", ha="center", va="center",
                 color="#7a2a1a", fontsize=8)
    for k in range(K):
        Qf = np.vstack([r["starts"][k], r["trajs_free"][k], r["goals"][k]])
        Qu = np.vstack([r["starts"][k], r["trajs"][k], r["goals"][k]])
        axT.plot(Qf[:, 1], Qf[:, 0], "--", color=BLIND, lw=1.3, alpha=0.55)
        axT.plot(Qu[:, 1], Qu[:, 0], "-", color=colors[k], lw=2.6)
        axT.scatter(r["starts"][k][1], r["starts"][k][0], color=colors[k], s=48, zorder=5)
        axT.scatter(r["goals"][k][1], r["goals"][k][0], color=colors[k], marker="*",
                    s=180, zorder=5)
    axT.plot([], [], "--", color=BLIND, lw=1.6, label="sans COSMOS (pousse à travers la dérive)")
    axT.plot([], [], "-", color=INK, lw=2.4, label="avec COSMOS (épouse la dérive orbitale)")
    axT.scatter([], [], color="#444", marker="o", s=48, label="départ")
    axT.scatter([], [], color="#444", marker="*", s=150, label="cible")
    axT.set_xlabel("le long de l'orbite  [unités normalisées]")
    axT.set_ylabel("radial")
    axT.set_aspect("equal"); axT.grid(alpha=0.25); axT.legend(fontsize=8.6, loc="upper right")
    axT.set_title("Reconfiguration d'une formation de 4 satellites", fontsize=11, fontweight="bold")

    # big callout
    axT.text(0.02, 0.03, f"−{r['saving']:.0f}% de carburant",
             transform=axT.transAxes, fontsize=19, fontweight="bold", color=FUEL,
             va="bottom", ha="left")
    axT.text(0.02, 0.005, "même mission, même sûreté", transform=axT.transAxes,
             fontsize=8.5, color=MUT, va="bottom", ha="left")

    # per-satellite fuel bars
    idx = np.arange(K); w = 0.38
    axB.bar(idx - w/2, r["perf"], w, color=BLIND, alpha=0.85, label="sans COSMOS")
    axB.bar(idx + w/2, r["per"], w, color=FUEL, label="avec COSMOS")
    axB.set_xticks(idx); axB.set_xticklabels([f"sat {i+1}" for i in range(K)], fontsize=9)
    axB.set_ylabel("Δv par satellite  (carburant)")
    axB.legend(fontsize=9); axB.grid(alpha=0.25, axis="y")
    axB.set_title(f"Total : {r['dvf']:.1f} → {r['dv']:.1f}   "
                  f"(sans collision, marge +{r['clr']:.1f})", fontsize=9.5)

    fig.suptitle("COSMOS — « lisser » dans la bonne métrique = dépenser moins de carburant",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    if verbose:
        print(f"=== HERO 2D ===  total Δv {r['dvf']:.2f} -> {r['dv']:.2f}  "
              f"(-{r['saving']:.0f}%), clearance +{r['clr']:.2f}")
        print(f"Figure -> {out}")
    return r


# ---------------------------------------------------------------------------
# HERO 3D : free vs fuel with cross-track + keep-out sphere + gain on plot
# ---------------------------------------------------------------------------
def _draw_3d(ax, r, elev=22, azim=-60):
    K = len(r["starts"]); colors = plt.cm.viridis(np.linspace(0.12, 0.85, K))
    for ob in r["obstacles"]:
        u, v = np.mgrid[0:2*np.pi:22j, 0:np.pi:14j]
        ax.plot_wireframe(ob["c"][1] + ob["r"]*np.cos(u)*np.sin(v),
                          ob["c"][0] + ob["r"]*np.sin(u)*np.sin(v),
                          ob["c"][2] + ob["r"]*np.cos(v),
                          color=KEEP, alpha=0.30, linewidth=0.5)
    for k in range(K):
        Qf = np.vstack([r["starts"][k], r["trajs_free"][k], r["goals"][k]])
        Qu = np.vstack([r["starts"][k], r["trajs"][k], r["goals"][k]])
        ax.plot(Qf[:, 1], Qf[:, 0], Qf[:, 2], "--", color=BLIND, lw=1.2, alpha=0.6)
        ax.plot(Qu[:, 1], Qu[:, 0], Qu[:, 2], "-", color=colors[k], lw=2.4)
        ax.scatter(r["starts"][k][1], r["starts"][k][0], r["starts"][k][2],
                   color=colors[k], s=36)
        ax.scatter(r["goals"][k][1], r["goals"][k][0], r["goals"][k][2],
                   color=colors[k], marker="*", s=140)
    ax.set_xlabel("le long de l'orbite"); ax.set_ylabel("radial"); ax.set_zlabel("hors-plan")
    ax.view_init(elev=elev, azim=azim)


def hero_3d(out="cosmos_hero_3d.png", verbose=True):
    r = _solve_both("debris", dim=3)
    fig = plt.figure(figsize=(9.5, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_3d(ax, r)
    ax.plot([], [], "--", color=BLIND, lw=1.6, label="sans COSMOS")
    ax.plot([], [], "-", color=INK, lw=2.4, label="avec COSMOS")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"COSMOS en 3D — rendez-vous hors-plan\n"
                 f"carburant total : {r['dvf']:.1f} → {r['dv']:.1f}   "
                 f"(−{r['saving']:.0f}%, sans collision)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    if verbose:
        print(f"=== HERO 3D ===  total Δv {r['dvf']:.2f} -> {r['dv']:.2f}  (-{r['saving']:.0f}%)")
        print(f"Figure -> {out}")
    return r


def make_gif(out="cosmos_hero_3d.gif", frames=48, verbose=True):
    """Short, light rotating GIF of the 3D hero (azimuth sweep)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    r = _solve_both("debris", dim=3, N=50)
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    def frame(i):
        ax.cla()
        _draw_3d(ax, r, elev=20, azim=-80 + 360 * i / frames)
        ax.set_title(f"COSMOS 3D — carburant −{r['saving']:.0f}% (sans collision)",
                     fontsize=10, fontweight="bold")
    anim = FuncAnimation(fig, frame, frames=frames, interval=60)
    anim.save(out, writer=PillowWriter(fps=16), dpi=80)
    plt.close(fig)
    if verbose:
        import os
        print(f"GIF -> {out}  ({os.path.getsize(out)//1024} KB)")


def main():
    hero_2d(); hero_3d()


if __name__ == "__main__":
    main()
