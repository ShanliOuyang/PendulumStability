# PendulumStability

A reinforcement learning project for stabilizing an inverted pendulum on a cart, developed during the **GEARS Summer Research Program** at **North Carolina State University (NCSU)**.

## RL Algorithm

<img width="897" height="524" alt="468073371-e31146f3-07d3-41da-aaae-12a7bb06329d" src="https://github.com/user-attachments/assets/7a22f1df-f061-47b2-a357-c8ffaac61e1c" />

## Overview

This project explores the application of **Proximal Policy Optimization (PPO)** to solve the cart-pole balancing problem with enhanced realism and generalization:

- **Custom Physics Environment**: Implements accurate motor dynamics including gear ratio, back-EMF, and viscous damping
- **Generalization Focus**: Expands the acceptable range of initial states to improve model robustness
- **Statistical Validation**: Uses Z-tests and confidence intervals to validate model improvements
- **Hardware Deployment**: Exports trained model parameters for MATLAB-based hardware implementation

## Project Structure

| File | Description |
|------|-------------|
| `myCartpoleF.py` | Custom CartPole environment with realistic physics (motor dynamics, RK4 integration) |
| `Train_Result.py` | PPO model training with early-stopping callback based on state stability |
| `Test_Result.py` | Large-scale testing (10,000 episodes) under varying initial conditions |
| `Validation.py` | Statistical comparison of model versions using Z-tests |
| `Turn-into-txt.ipynb` | Export model parameters (weights/biases) for MATLAB deployment |
| `models/` | Trained best model (`.zip`) |
| `Text_Results/` | Exported model parameters in text format |
| `Training/` | Training logs and intermediate model checkpoints |

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

- Create a custom CartPole environment with realistic physics
- Train a PPO agent with early stopping when the pendulum stabilizes
- Save the best model to `Training/Saved Models/best_model.zip`

**2. Test the Model**

```bash
python Test_Result.py
```

This will:

- Run 10,000 test episodes with varying initial conditions
- Dynamically expand the initial state range to test generalization
- Save results to `Episode_State/episode_results_range_*.csv`

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

## Results

| Metric | Value |
|--------|-------|
| Success Rate (within training distribution) | ~95% |
| Generalization Range (tested) | ±0.20 rad |
| Average Episode Length | 2000+ steps |

## Related Imformation

- **Shanli Ouyang** - NCSU GEARS Program (with group members)
- **Dr. Hien Tran** for guidance and mentorship



