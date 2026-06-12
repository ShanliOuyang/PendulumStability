<img width="897" height="524" alt="image" src="https://github.com/user-attachments/assets/9c4ea386-6c42-4c9a-944e-7dc17eb3ac31" /># PendulumStability

A reinforcement learning project for stabilizing an inverted pendulum on a cart, developed during the **GEARS Summer Research Program** at **North Carolina State University (NCSU)**.

## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [Core Features](#core-features)
- [Project Structure](#project-structure)
- [Algorithms and Methods](#algorithms-and-methods)
- [Getting Started](#getting-started)
- [Methodology](#methodology)
- [Workflow](#workflow)
- [Physics Background](#physics-background)
- [Hardware Deployment](#hardware-deployment)
- [Outputs](#outputs)
- [Results](#results)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Overview

This project explores the application of **Proximal Policy Optimization (PPO)** to solve the cart-pole balancing problem with enhanced realism and generalization:

- **Custom Physics Environment**: Implements accurate motor dynamics including gear ratio, back-EMF, and viscous damping
- **Generalization Focus**: Expands the acceptable range of initial states to improve model robustness
- **Statistical Validation**: Uses Z-tests and confidence intervals to validate model improvements
- **Hardware Deployment**: Exports trained model parameters for MATLAB-based hardware implementation

<img width="897" height="524" alt="image" src="https://github.com/user-attachments/assets/fa65190e-a743-4110-84c0-d49d5c505ee2" />

## Core Features

- **Custom continuous-control CartPole environment** built on Gymnasium
- **Physics-aware dynamics model** including motor voltage input, gearbox ratio, viscous damping, and pendulum-cart coupling
- **Continuous action space** where the policy outputs normalized motor commands that are scaled into actuator force
- **PPO-based policy training** with Stable-Baselines3
- **Early stopping based on state stability** using custom callbacks that monitor cart position, cart velocity, pole angle, and pole angular velocity
- **Robustness testing under wider reset ranges** to evaluate generalization outside the default training distribution
- **Statistical model comparison** using large-sample proportion testing and Wilson confidence intervals
- **Parameter export pipeline** for downstream MATLAB or hardware-side inference
- **Training logs and saved checkpoints** for model tracking and reuse

## Project Structure

| File | Description |
|------|-------------|
| `myCartpoleF.py` | Custom CartPole environment with actuator-aware physics and rendering support |
| `Train_Result.py` | PPO model training with early-stopping callback based on state stability |
| `Test_Result.py` | Large-scale testing (10,000 episodes) under varying initial conditions |
| `Validation.py` | Statistical comparison of model versions using Z-tests |
| `Turn-into-txt.ipynb` | Export model parameters (weights/biases) for MATLAB deployment |
| `models/` | Trained best model (`.zip`) |
| `Text_Results/` | Exported model parameters in text format |
| `Training/` | Training logs and intermediate model checkpoints |

## Algorithms and Methods

### Reinforcement Learning Algorithm

- **Algorithm**: Proximal Policy Optimization (**PPO**)
- **Implementation**: `stable_baselines3.PPO`
- **Policy Type**: `MlpPolicy`
- **Action Type**: continuous 1D action in `[-1, 1]`, scaled to motor force

### Environment Modeling

The environment in `myCartpoleF.py` extends the classic CartPole setting by replacing the standard simplified force model with a more realistic actuator-aware dynamics model. The implementation includes:

- cart-pole nonlinear dynamics
- motor torque and back-EMF terms
- gearbox ratio and pinion radius
- equivalent viscous damping at the motor and pendulum axis
- configurable reset ranges for the initial state

### Numerical Update Strategy

The code uses a high-frequency simulation step (`100 Hz`) and updates the state with the environment's custom physics routine. The script labels the integrator as `RK4`, while the actual state update path in the current code follows a semi-implicit Euler-style update sequence.

### Early-Stopping Logic

`Train_Result.py` defines two custom callback classes:

- `StableWindow`: tracks whether one state signal stays within a tolerance window for a fixed number of steps
- `AndCallback`: combines multiple `StableWindow` instances and stops training when all monitored signals are stable at the same time

This makes training stop based on behavioral stability rather than only a reward threshold.

## Getting Started

### Prerequisites

```bash
pip install gymnasium stable-baselines3[extra] matplotlib statsmodels pygame jupyter
```

### Quick Start

**1. Train a Model**

```bash
python Train_Result.py
```

This will:

- Register the custom `CartPoleLab` environment
- Run a short environment sanity check with rendering
- Train a PPO agent with callback-based early stopping
- Save evaluation-selected and final models to the `models/` and `Training/Saved Models/` folders

**2. Test the Model**

```bash
python Test_Result.py
```

This will:

- Run 10,000 test episodes with varying initial conditions
- Dynamically expand the initial state range to test generalization
- Save episode-level initial-state labels to `Episode_State/episode_results_range_*.csv`

**3. Validate Model Performance**

```bash
python Validation.py
```

This will:

- Compare "old" and "new" models using 100,000 test episodes
- Perform Z-test for statistical significance
- Output confidence intervals and improvement recommendations

**4. Export for Hardware Deployment**

Open `Turn-into-txt.ipynb` in Jupyter and run all cells to export model parameters to `Text_Results/`.

## Methodology

### Custom Environment

The `myCartpoleF.py` extends Gymnasium's CartPole with:

1. **Realistic Motor Dynamics** with parameters from Emi's thesis:
   - Motor pinion radius: 6.35e-3 m
   - Rotor moment of inertia: 3.90e-7 kg·m²
   - Planetary gearbox ratio: 3.71
   - Motor armature resistance: 2.6 Ω
   - Motor torque constant: 0.00767 N·m/A

2. **Numerical Integration** (RK4 or Semi-Implicit Euler)

3. **Flexible Reset Bounds** for domain randomization

### Training Strategy

- **Algorithm**: PPO with MLP policy
- **Early Stopping**: Custom `StableWindow` callback monitors state stability
- **Key Hyperparameters**:
  - `n_steps = 4096`
  - `batch_size = 256`
  - `learning_rate = 3e-3` (linear decay)
  - `clip_range = 0.15`

### Generalization Testing

Tests model robustness by gradually expanding initial state ranges:

```
Initial Range: ±0.08 → ±0.10 → ±0.12 → ... → ±0.20
```

Success criteria:

- Balance rate ≥ 40%
- Episode length ≥ 1000 steps

### Statistical Validation

Uses **proportions Z-test** to compare model versions:

- **H₀**: New model success rate ≤ Old model success rate
- **H₁**: New model success rate > Old model success rate
- **Significance level**: α = 0.05
- **Confidence intervals**: Wilson method

## Workflow

The repository is organized around a practical training-to-validation workflow:

1. `myCartpoleF.py` defines the custom environment and the physical system dynamics.
2. `Train_Result.py` registers the environment, trains a PPO policy, evaluates it during training, and saves model checkpoints.
3. `Test_Result.py` reloads the trained model and stress-tests it under progressively larger reset ranges.
4. `Validation.py` compares two saved model versions statistically using repeated rollout experiments.
5. `Turn-into-txt.ipynb` exports trained network parameters into plain-text files for external use.

## Physics Background

The equations of motion are derived from Lagrangian mechanics with motor dynamics.

**State Vector**: `[x, x_dot, theta, theta_dot]` (cart position, velocity, pole angle, angular velocity)

**Termination Conditions**:

- `|theta| > 0.2 rad` (~11.5°)
- `|x| > 0.2 m`
- Episode length > 2000 steps

## Hardware Deployment

The trained model can be deployed to physical hardware:

1. **Export Parameters**: Run `Turn-into-txt.ipynb`
2. **Generated Files**:
   - `W0.txt`, `b0.txt` - Layer 1 weights and biases
   - `W1.txt`, `b1.txt` - Layer 2 weights and biases
   - `W_out.txt`, `b_out.txt` - Output layer weights and biases
3. **MATLAB Integration**: Load text files and implement forward pass

## Outputs

During normal use, the project can generate the following artifacts:

- **Saved models**: PPO checkpoints in `models/` and `Training/Saved Models/`
- **Training logs**: TensorBoard-compatible logs in `Training/Logs/`
- **Episode test CSVs**: success labels and initial states for generalization tests
- **Text parameter dumps**: weight and bias matrices in `Text_Results/`
- **Interactive visualization**: Pygame rendering for environment inspection and evaluation

## Results

| Metric | Value |
|--------|-------|
| Success Rate (within training distribution) | ~95% |
| Generalization Range (tested) | ±0.20 rad |
| Average Episode Length | 2000+ steps |

## Authors

- **Shanli Ouyang** - *Initial work* - NCSU GEARS Program
- **Dr. Hien Tran** for guidance and mentorship
