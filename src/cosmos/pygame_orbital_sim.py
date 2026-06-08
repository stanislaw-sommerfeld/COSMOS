"""
COSMOS - Real-time orbital simulation (Pygame)
==============================================

A real-time, interactive window (like a boids sim) showing K satellites that
follow their CHOMP-planned trajectories around the Earth (a keep-out zone),
while staying clear of each other.

The planner (solve_swarm, Part II) computes collision-free trajectories ONCE;
the loop then plays them back in real time. F_obs = Earth, F_agents = satellites.

Controls:
  SPACE  pause / resume
  R      re-plan a new random configuration
  ESC/Q  quit

Run locally:
  python3 pygame_orbital_sim.py

Note: free-space kinematics dressed as an orbital scene. Real Clohessy-Wiltshire
dynamics is Part III; this file is the live visualization shell.
"""

import os
import numpy as np
import pygame

from .chomp_swarm import solve_swarm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
W, H = 760, 760
FPS = 60
K = 5
R_EARTH = 2.5
R_RING = 7.0
N_WP = 60
WORLD_EXTENT = 9.5                      # world units from center to edge
SCALE = (W / 2) / WORLD_EXTENT
CX, CY = W / 2, H / 2

BG = (5, 6, 10)
EARTH_COL = (31, 111, 235)
EARTH_RING = (58, 139, 255)
WHITE = (235, 238, 245)
GREY = (120, 124, 135)

# satellite palette (turbo-ish, distinct)
PALETTE = [(64, 200, 255), (120, 230, 120), (255, 196, 64),
           (255, 110, 90), (200, 120, 255)]


def w2s(p):
    """World -> screen pixels."""
    return int(CX + p[0] * SCALE), int(CY - p[1] * SCALE)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def plan(seed):
    """Generate a random start/target config and plan collision-free paths."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False) + phase
    starts = [np.array([R_RING * np.cos(a), R_RING * np.sin(a)]) for a in ang]
    # targets: antipodal + a random twist so configs vary
    twist = rng.uniform(0.6, 1.0) * np.pi
    goals = [np.array([R_RING * np.cos(a + twist), R_RING * np.sin(a + twist)])
             for a in ang]
    earth = [{"c": np.array([0.0, 0.0]), "r": R_EARTH}]
    trajs, _, _ = solve_swarm(
        starts, goals, obstacles=earth, strategy="gauss_seidel",
        n=N_WP, dt=1.0, lam=0.6, mu=1.2, eps_o=0.9, eps_a=1.0,
        r_safe=0.9, step=0.02, iters=350, seed=seed)
    paths = [np.vstack([starts[k], trajs[k], goals[k]]) for k in range(K)]
    # metrics
    earth_clr = min(np.min(np.linalg.norm(p, axis=1)) - R_EARTH for p in paths)
    inter = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            inter = min(inter, np.min(np.linalg.norm(paths[i] - paths[j], axis=1)))
    return paths, goals, earth_clr, inter


def sample(path, s):
    """Position along a path at fractional progress s in [0, 1]."""
    idx = s * (len(path) - 1)
    i0 = min(int(np.floor(idx)), len(path) - 2)
    f = idx - i0
    return (1 - f) * path[i0] + f * path[i0 + 1]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_scene(screen, font, paths, goals, s, earth_clr, inter, paused):
    screen.fill(BG)

    # planned paths (faint)
    for k in range(K):
        pts = [w2s(p) for p in paths[k]]
        pygame.draw.lines(screen, tuple(int(c * 0.30) for c in PALETTE[k]),
                          False, pts, 1)

    # Earth
    pygame.draw.circle(screen, EARTH_COL, (int(CX), int(CY)), int(R_EARTH * SCALE))
    pygame.draw.circle(screen, EARTH_RING, (int(CX), int(CY)),
                       int((R_EARTH + 0.18) * SCALE), 1)
    label = font.render("Earth", True, WHITE)
    screen.blit(label, label.get_rect(center=(CX, CY)))

    # targets
    for k in range(K):
        gx, gy = w2s(goals[k])
        pygame.draw.line(screen, PALETTE[k], (gx - 6, gy - 6), (gx + 6, gy + 6), 2)
        pygame.draw.line(screen, PALETTE[k], (gx - 6, gy + 6), (gx + 6, gy - 6), 2)

    # satellites + trails
    TRAIL = 22
    for k in range(K):
        lo = max(0.0, s - TRAIL / len(paths[k]))
        ss = np.linspace(lo, s, TRAIL)
        tp = [w2s(sample(paths[k], u)) for u in ss]
        if len(tp) > 1:
            pygame.draw.lines(screen, PALETTE[k], False, tp, 2)
        pos = w2s(sample(paths[k], s))
        pygame.draw.circle(screen, PALETTE[k], pos, 7)
        pygame.draw.circle(screen, WHITE, pos, 7, 1)

    # HUD
    lines = [
        f"COSMOS - orbital transfer   t = {s*100:3.0f}%   {'[PAUSED]' if paused else ''}",
        f"min Earth clearance: {earth_clr:+.2f}    min inter-sat: {inter:+.2f}",
        "SPACE pause   R re-plan   ESC quit",
    ]
    for i, t in enumerate(lines):
        col = WHITE if i == 0 else GREY
        screen.blit(font.render(t, True, col), (16, 14 + i * 22))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(headless=False, shot_path=None):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("COSMOS - orbital simulation")
    font = pygame.font.SysFont("consolas,menlo,monospace", 16)
    clock = pygame.time.Clock()

    seed = 3
    paths, goals, earth_clr, inter = plan(seed)
    s = 0.0
    speed = 0.18          # progress per second
    paused = False
    running = True
    frames = 0

    while running:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    seed += 1
                    paths, goals, earth_clr, inter = plan(seed)
                    s = 0.0

        if not paused:
            s += speed * dt
            if s >= 1.0:
                s = 0.0                      # loop the playback

        draw_scene(screen, font, paths, goals, s, earth_clr, inter, paused)
        pygame.display.flip()

        frames += 1
        if headless and frames >= 40:
            if shot_path:
                pygame.image.save(screen, shot_path)
            running = False

    pygame.quit()


if __name__ == "__main__":
    if os.environ.get("HEADLESS"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        main(headless=True, shot_path="pygame_sim_screenshot.png")
        print("headless OK -> pygame_sim_screenshot.png")
    else:
        main()
