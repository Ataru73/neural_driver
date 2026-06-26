# 2D Autonomous Driving Simulator with PyTorch

This project implements a 2D car racing simulator in Python where an agent learns to navigate a custom racetrack in the shortest time possible without going off-track. The project features two distinct training pathways: **Deep Q-Learning (DQN)** and a **Genetic Algorithm (Neuroevolution)**, built using **PyTorch**, **Gymnasium**, and **Pygame**.

The track centerline is generated dynamically using smooth **Catmull-Rom spline interpolation** from a set of control points, ensuring smooth sweeping curves and natural road layouts.

---

## Project Structure

The project is divided into the following key files:

### Core Files & DQN
- **`environment.py`**: Implements the racetrack and the Gymnasium-compatible simulator (`CarRacingEnv`) for a single car. Handles vehicle physics, distance raycasting (7 laser sensors), progress/checkpoint tracking, and rendering.
- **`model.py`**: Defines the PyTorch Neural Network architecture (3-layer Feed-Forward Network with ReLU activations) shared by both DQN and the Genetic Algorithm.
- **`agent.py`**: Implements the DQN Agent, including action selection (epsilon-greedy), the experience replay buffer (`ReplayBuffer`), learning updates, and robust model saving/loading.
- **`train.py`**: Headless training script for the DQN agent. Automatically saves the absolute best model to `models/best_model.pt` and exports a progress graph to `models/training_progress.png`.
- **`enjoy.py`**: Interactive visualization tool. Loads any trained model (DQN or Genetic) and lets you watch the AI drive, or allows you to take over manually using arrow keys or WASD.

### Genetic Algorithm (Neuroevolution)
- **`genetic_environment.py`**: Extends the single-car simulator into a multi-agent environment (`GeneticCarRacingEnv`) that simulates a population of cars racing simultaneously in the same environment.
- **`train_genetic.py`**: Implements the Genetic Algorithm training loop. Features elitism, uniform crossover, Gaussian mutations, and visual rendering during evolution.

---

## Installation & Setup

1. **Create and activate the virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Deep Q-Learning (DQN)

#### Train a DQN Agent (Headless/Fast):
```bash
python train.py
```
*Note: This runs in headless mode to maximize frames per second. Training metrics (average reward, checkpoints reached, current epsilon) are printed every 10 episodes.*

#### Resume DQN Training:
You can resume training from a saved checkpoint or final model while preserving the exploration rate ($\epsilon$) and optimizer state:
```bash
python train.py models/final_model.pt
```

---

### 2. Genetic Algorithm (Neuroevolution)

The Genetic Algorithm simulates a population of 40 cars simultaneously. At each generation, slow or crashed cars are eliminated. The top performers (highest fitness) are selected to reproduce via crossover and mutation.

When a car successfully completes 1 lap, its simulation is immediately stopped, and its final fitness is calculated based on its completion time:
$$\text{Fitness} = 50000.0 - (\text{total\_steps} \times 5.0)$$
This ensures that shorter lap times (fewer steps) result in higher fitness, forcing the population to evolve faster driving behaviors.

#### Train Headless (Fast):
```bash
python train_genetic.py
```

#### Train Visually (Highly Recommended!):
Watch the entire population learn, crash, and race in real-time on your screen:
```bash
python train_genetic.py --render
```
- *Crashed or stagnant cars turn semi-transparent red, while the current generation leader is highlighted in cyan, projecting its active laser sensors.*

#### Set the Number of Generations:
By default, the Genetic Algorithm runs for 100 generations. You can change this using the `--generations=N` parameter:
```bash
python train_genetic.py --generations=50
```

#### Resume Genetic Training from a Pretrained Model:
You can seed a new genetic population using a pretrained champion model (from either DQN or GA):
```bash
python train_genetic.py models/genetic_best.pt
# or with visualization:
python train_genetic.py models/genetic_best.pt --render
```
*The algorithm clones the champion exactly as the elite agent (to preserve its skills) and fills the rest of the population with mutated variations of it (using mixed mutation scales to explore nearby trajectories).*

---

### 3. Visualizing a Trained Model or Driving Manually

To watch a trained model drive, or to drive the car yourself:
```bash
python enjoy.py models/genetic_best.pt  # or models/best_model.pt
```

#### Key Controls:
- **`A`**: Switch to **AI Mode** (the neural network takes control).
- **`M`**: Switch to **Manual Mode** (you drive the car).
- **`R`**: Reset the track and place the car back at the starting line.
- **`ESC`**: Exit the simulator.

#### Driving Controls (Manual Mode):
- **Up Arrow / W**: Accelerate.
- **Down Arrow / S**: Brake / Reverse.
- **Left Arrow / A**: Steer Left.
- **Right Arrow / D**: Steer Right.

---

## Fixed Starting Position

By default, training episodes randomly spawn the car at different checkpoints along the racetrack. This helps the network learn to recover from various configurations and corners. However, if you want the car to always spawn at the main starting line (checkpoint 0), you can suppress the starting-position randomness by passing the `--fixed-start` argument:

* **Train DQN with a fixed starting position**:
  ```bash
  python train.py --fixed-start
  ```
* **Train Genetic Algorithm with a fixed starting position visually**:
  ```bash
  python train_genetic.py --fixed-start --render
  ```

---

## Racetrack Obstacles

You can inject random obstacles onto the racetrack to make the driving task more challenging and force the networks to learn obstacle-avoidance maneuvers. Obstacles are drawn as circular striped traffic drums (red/white) and are placed dynamically within the road boundaries at random checkpoints (excluding the starting zone).

To train or enjoy with obstacles, pass the `--obstacles=N` argument (where `N` is the number of obstacles):

* **Train DQN with 5 obstacles**:
  ```bash
  python train.py --obstacles=5
  ```
* **Train Genetic Algorithm with 10 obstacles visually**:
  ```bash
  python train_genetic.py --obstacles=10 --render
  ```
* **Watch or drive with 8 obstacles**:
  ```bash
  python enjoy.py models/genetic_best.pt --obstacles=8
  ```

---

## Technical Details


- **Sensor Inputs (State Space)**: 11 normalized distance measurements from laser raycasts angled at `[-90°, -70°, -45°, -30°, -15°, 0°, 15°, 30°, 45°, 70°, 90°]` relative to the car's heading, plus the car's normalized current speed.
- **Short-Term Memory (Frame Stacking)**: To help the network handle obstacles and coordinate complex maneuvers, **Frame Stacking** is implemented. The environment buffers the 6 most recent raw observations and concatenates them sequentially. This expands the state dimension to 138 (23 inputs $\times$ 6 stacked frames), providing the neural network with temporal context. This allows the model to implicitly compute velocity, rates of approach to obstacles/boundaries, and acceleration.
- **Action Space**: 9 discrete actions representing combinations of steering (Left, Straight, Right) and acceleration (Accelerate, Coast, Brake).
- **DQN Reward Design**:
  - Base time penalty (`-0.1` per step) to encourage speed.
  - Positive reward (`+15`) for hitting sequential checkpoints in the correct order.
  - Alignment reward based on the cosine of the heading angle error compared to the track centerline tangent.
  - Collision penalty (`-100`) for hitting walls.

---

## Street Lamp (Traffic Signal)

To simulate intersection or traffic signal navigation, a **Street Lamp (Traffic Signal)** is placed dynamically on the racetrack:
* **Position**: Set randomly along the outer track border of a centerline checkpoint during `reset()`.
* **Stop Line**: A solid white line is drawn across the width of the racetrack at the lamp's checkpoint.
* **Alternating Colors**: The lamp color cycles dynamically: Green $\rightarrow$ Yellow $\rightarrow$ Red $\rightarrow$ Green every 60 steps.
* **Collision Check**: If the lamp is **Red**, cars must stop before the stop line. If any part of the car crosses the line (`dist_long > -12.0`), it is treated as **crashed/collided**. Once a car successfully crosses the line while it is **Green/Yellow**, it can proceed freely.
* **Sensor Visibility Constraint (DQN Input)**:
  - The car can detect the lamp's state (color one-hot, relative distance, and relative angle) **only if** the lamp checkpoint is in front of the car (`cos(rel_angle) >= 0`) and within the maximum sensor/laser distance (`<= 250.0`).
  - Outside of this range, the observation values are zeroed out (lamp color set to `[0, 0, 0]`, distance to `1.0`, and angle to `0.0`).
