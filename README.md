# Swarm-Based Path Planning with Obstacles using PSO

## Student Information

- **Name:** ANUM ZARA
- **Roll Number:** 01-136232-098
- **Seed:** 01136232098
- **Course:** Swarm Intelligence
- **Assignment:** Assignment 1

## Objective

This project implements Particle Swarm Optimization (PSO) to search for a short, obstacle-free path from a randomly generated start point to a randomly generated goal point on a 2D grid.

The problem instance is generated from the student's roll number. Therefore, different roll numbers produce different obstacles, start points, and goal points.

## Approach

1. Use the roll number as the random seed.
2. Generate a 20×20 grid.
3. Randomly generate obstacle cells.
4. Randomly select free start and goal cells.
5. Represent each PSO particle as a sequence of intermediate 2-D waypoints.
6. Construct a complete path from start → waypoints → goal.
7. Calculate path cost using:
   - total path length,
   - a large penalty for obstacle collisions,
   - a penalty for out-of-bound points,
   - a small penalty for repeated cells.
8. Update particles using PSO velocity and position equations.
9. Keep the best solution found by the swarm.
10. Visualize the grid, obstacles, start, goal, final path, and convergence.

## PSO Parameters

- Grid size: 20 × 20
- Particles: 40
- Iterations: 250
- Intermediate waypoints: 8
- Inertia weight (W): 0.72
- Cognitive coefficient (C1): 1.45
- Social coefficient (C2): 1.45

## How to Run

Install the required packages:

```bash
pip install numpy matplotlib
```

Run:

```bash
python path_planning.py
```

Enter your roll number when prompted.

Example:

```text
Enter your roll number: YOUR_ROLL_NUMBER
```

The program generates:

- `output/best_path.png`
- `output/convergence.png`
- `output/result.txt`

## Important

The roll number must be used as the seed because the assignment requires every student to have a unique problem instance.

## Flow Diagram

**Draw this flow by hand on paper and take a clear photograph. Then place the photo here in the README.**

```text
START
  |
  v
Enter Roll Number
  |
  v
Set Random Seed = Roll Number
  |
  v
Generate Grid
  |
  v
Generate Obstacles
  |
  v
Generate Free Start and Goal
  |
  v
Initialize PSO Particles
  |
  v
Generate Candidate Paths
  |
  v
Calculate Path Cost
  |
  v
Check Obstacle Collision
  |
  v
Update Personal Best
  |
  v
Update Global Best
  |
  v
Update Velocity and Position
  |
  v
Stopping Condition?
  |---------------- No ----------------|
  |                                    |
  |<-----------------------------------|
  |
 Yes
  |
  v
Output Best Path and Cost
  |
  v
Visualize Grid and Path
  |
  v
END
```

## Suggested GitHub Commit History

Make at least 5 meaningful commits:

```text
1. Initial project setup
2. Add random grid and obstacle generation
3. Implement PSO particle and fitness functions
4. Add PSO optimization and collision handling
5. Add visualization and convergence graph
6. Add README and final results
```

Do not upload everything as one final commit.

## Submission

Submit the public GitHub repository link according to the assignment instructions.
