"""
Swarm Intelligence - Assignment 1
Swarm-Based Path Planning with Obstacles using PSO

The student's roll number is used as the random seed.
"""

import random
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. STUDENT SETTINGS
# ============================================================

ROLL_NUMBER = int(input("Enter your roll number: "))

# The assignment requires the roll number to be the random seed.
SEED = ROLL_NUMBER
random.seed(SEED)
np.random.seed(SEED)

# Grid and obstacle settings are generated from the roll-number seed
GRID_SIZE = random.randint(15, 25)
OBSTACLE_PROBABILITY = random.uniform(0.15, 0.25)

NUM_PARTICLES = 40
MAX_ITERATIONS = 250

# Number of intermediate waypoints in each particle.
NUM_WAYPOINTS = 8

# PSO parameters
W = 0.72
C1 = 1.45
C2 = 1.45

# Cost penalties
COLLISION_PENALTY = 1000.0
OUT_OF_BOUNDS_PENALTY = 1000.0
REPEATED_CELL_PENALTY = 5.0

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. GENERATE UNIQUE PROBLEM INSTANCE
# ============================================================

def generate_problem():
    """
    Generate obstacles, start, and goal from the roll-number seed.
    Start and goal are always placed on free cells.
    """

    all_cells = [
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
    ]

    start = random.choice(all_cells)
    remaining = [p for p in all_cells if p != start]
    goal = random.choice(remaining)

    obstacles = set()

    for cell in all_cells:
        if cell == start or cell == goal:
            continue

        if random.random() < OBSTACLE_PROBABILITY:
            obstacles.add(cell)

    return start, goal, obstacles


START, GOAL, OBSTACLES = generate_problem()


# ============================================================
# 3. BASIC GRID FUNCTIONS
# ============================================================

def clamp_point(point):
    """Keep a continuous point inside the grid."""
    x = np.clip(point[0], 0, GRID_SIZE - 1)
    y = np.clip(point[1], 0, GRID_SIZE - 1)
    return np.array([x, y], dtype=float)


def point_to_cell(point):
    """Convert a continuous waypoint to the nearest grid cell."""
    x = int(round(point[0]))
    y = int(round(point[1]))

    x = max(0, min(GRID_SIZE - 1, x))
    y = max(0, min(GRID_SIZE - 1, y))

    return (x, y)


def segment_cells(p1, p2):
    """
    Return grid cells touched by a line segment.
    Uses dense interpolation so obstacle crossings are detected.
    """
    distance = np.linalg.norm(np.array(p2) - np.array(p1))
    steps = max(2, int(math.ceil(distance * 8)))

    cells = []
    for t in np.linspace(0, 1, steps + 1):
        p = np.array(p1) * (1 - t) + np.array(p2) * t
        cells.append(point_to_cell(p))

    # Remove consecutive duplicates
    unique = []
    for cell in cells:
        if not unique or cell != unique[-1]:
            unique.append(cell)

    return unique


# ============================================================
# 4. PARTICLE REPRESENTATION
# ============================================================

def make_random_particle():
    """
    A particle contains NUM_WAYPOINTS intermediate 2-D waypoints.
    Start and goal are fixed and are not part of the particle vector.
    """

    points = np.random.uniform(
        0,
        GRID_SIZE - 1,
        size=(NUM_WAYPOINTS, 2)
    )

    return points


def build_path(position):
    """Build complete path: start -> intermediate waypoints -> goal."""
    points = [np.array(START, dtype=float)]
    points.extend(position)
    points.append(np.array(GOAL, dtype=float))
    return points


# ============================================================
# 5. FITNESS / COST FUNCTION
# ============================================================

def path_cost(position):
    """
    Lower cost is better.

    Cost =
        total path length
        + collision penalties
        + boundary penalties
        + repeated-cell penalties
    """

    points = build_path(position)

    total_length = 0.0
    collision_count = 0
    out_of_bounds_count = 0
    repeated_count = 0

    visited = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        total_length += float(np.linalg.norm(p2 - p1))

        cells = segment_cells(p1, p2)

        for cell in cells:
            visited.append(cell)

            if cell in OBSTACLES:
                collision_count += 1

    # Penalize waypoint coordinates outside the grid.
    for p in position:
        if (
            p[0] < 0 or p[0] > GRID_SIZE - 1 or
            p[1] < 0 or p[1] > GRID_SIZE - 1
        ):
            out_of_bounds_count += 1

    # Repeated cells are discouraged because they usually indicate
    # unnecessary loops.
    repeated_count = len(visited) - len(set(visited))

    cost = (
        total_length
        + COLLISION_PENALTY * collision_count
        + OUT_OF_BOUNDS_PENALTY * out_of_bounds_count
        + REPEATED_CELL_PENALTY * repeated_count
    )

    return cost


def collision_count(position):
    """Count obstacle cells touched by the path."""
    points = build_path(position)
    count = 0

    for i in range(len(points) - 1):
        cells = segment_cells(points[i], points[i + 1])
        count += sum(cell in OBSTACLES for cell in cells)

    return count


# ============================================================
# 6. PARTICLE SWARM OPTIMIZATION
# ============================================================

def run_pso():
    """
    Standard PSO over continuous intermediate waypoint coordinates.
    """

    positions = np.array([
        make_random_particle()
        for _ in range(NUM_PARTICLES)
    ])

    velocities = np.random.uniform(
        -1.0,
        1.0,
        size=positions.shape
    )

    personal_best_positions = positions.copy()

    personal_best_costs = np.array([
        path_cost(p) for p in positions
    ])

    best_index = np.argmin(personal_best_costs)

    global_best_position = personal_best_positions[best_index].copy()
    global_best_cost = personal_best_costs[best_index]

    history = [global_best_cost]

    for iteration in range(MAX_ITERATIONS):

        r1 = np.random.random(size=positions.shape)
        r2 = np.random.random(size=positions.shape)

        velocities = (
            W * velocities
            + C1 * r1 * (personal_best_positions - positions)
            + C2 * r2 * (global_best_position - positions)
        )

        # Limit velocity to keep movement stable.
        velocities = np.clip(velocities, -2.5, 2.5)

        positions = positions + velocities

        # Keep all particles inside the grid.
        positions = np.clip(
            positions,
            0,
            GRID_SIZE - 1
        )

        current_costs = np.array([
            path_cost(p) for p in positions
        ])

        improved = current_costs < personal_best_costs

        personal_best_positions[improved] = positions[improved]
        personal_best_costs[improved] = current_costs[improved]

        best_index = np.argmin(personal_best_costs)

        if personal_best_costs[best_index] < global_best_cost:
            global_best_cost = personal_best_costs[best_index]
            global_best_position = (
                personal_best_positions[best_index].copy()
            )

        history.append(global_best_cost)

        if collision_count(global_best_position) == 0:
            # Continue a few more iterations to improve path length.
            pass

    return global_best_position, global_best_cost, history


# ============================================================
# 7. PATH CLEANING
# ============================================================

def path_cells(position):
    """Convert final continuous path to grid cells."""
    points = build_path(position)
    cells = []

    for i in range(len(points) - 1):
        cells.extend(segment_cells(points[i], points[i + 1]))

    cleaned = []
    for cell in cells:
        if not cleaned or cell != cleaned[-1]:
            cleaned.append(cell)

    return cleaned


def has_collision(cells):
    return any(cell in OBSTACLES for cell in cells)


# ============================================================
# 8. VISUALIZATION
# ============================================================

def plot_solution(best_position, best_cost, history):
    cells = path_cells(best_position)

    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Obstacles
    if OBSTACLES:
        ox = [p[0] for p in OBSTACLES]
        oy = [p[1] for p in OBSTACLES]

        ax.scatter(
            ox,
            oy,
            marker="s",
            s=100,
            label="Obstacle"
        )

    # Final path
    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        label="PSO Best Path"
    )

    # Start and goal
    ax.scatter(
        START[0],
        START[1],
        marker="o",
        s=180,
        label="Start"
    )

    ax.scatter(
        GOAL[0],
        GOAL[1],
        marker="*",
        s=250,
        label="Goal"
    )

    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_title(
        f"PSO Swarm-Based Path Planning | Seed={SEED}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    path_file = OUTPUT_DIR / "best_path.png"
    fig.savefig(path_file, dpi=200)
    plt.show()

    # Convergence graph
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("PSO Convergence")
    plt.grid(True, alpha=0.3)

    convergence_file = OUTPUT_DIR / "convergence.png"
    plt.savefig(convergence_file, dpi=200, bbox_inches="tight")
    plt.show()

    return path_file, convergence_file


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("SWARM INTELLIGENCE - ASSIGNMENT 1")
    print("PSO Swarm-Based Path Planning with Obstacles")
    print("=" * 60)

    print(f"Seed / Roll Number : {SEED}")
    print(f"Grid Size          : {GRID_SIZE} x {GRID_SIZE}")
    print(f"Start              : {START}")
    print(f"Goal               : {GOAL}")
    print(f"Number of Obstacles: {len(OBSTACLES)}")
    print()

    print("Running PSO...")

    best_position, best_cost, history = run_pso()

    cells = path_cells(best_position)
    collision = has_collision(cells)

    # Geometric path length
    path_length = 0.0
    points = build_path(best_position)

    for i in range(len(points) - 1):
        path_length += float(
            np.linalg.norm(points[i + 1] - points[i])
        )

    print()
    print("-" * 60)
    print("FINAL RESULT")
    print("-" * 60)

    print(f"Best Cost       : {best_cost:.4f}")
    print(f"Path Length     : {path_length:.4f}")
    print(f"Path Cells      : {len(cells)}")
    print(f"Obstacle Hits   : {collision_count(best_position)}")
    print(f"Collision-Free  : {not collision}")
    print()

    print("Path:")
    print(cells)

    if collision:
        print()
        print(
            "WARNING: PSO did not find a collision-free path "
            "in this run. Increase MAX_ITERATIONS or NUM_PARTICLES."
        )
    else:
        print()
        print("SUCCESS: Collision-free path found.")

    plot_solution(
        best_position,
        best_cost,
        history
    )

    # Save a text result
    result_file = OUTPUT_DIR / "result.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("SWARM INTELLIGENCE - ASSIGNMENT 1\n")
        f.write("PSO Swarm-Based Path Planning\n\n")
        f.write(f"Seed / Roll Number: {SEED}\n")
        f.write(f"Grid Size: {GRID_SIZE} x {GRID_SIZE}\n")
        f.write(f"Start: {START}\n")
        f.write(f"Goal: {GOAL}\n")
        f.write(f"Number of Obstacles: {len(OBSTACLES)}\n")
        f.write(f"Best Cost: {best_cost:.4f}\n")
        f.write(f"Path Length: {path_length:.4f}\n")
        f.write(f"Obstacle Hits: {collision_count(best_position)}\n")
        f.write(f"Collision-Free: {not collision}\n")
        f.write(f"Path: {cells}\n")


if __name__ == "__main__":
    main()
