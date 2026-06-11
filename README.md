# Pendulum Stability - Reinforcement Learning for Cart-Pole System

![Cart-Pole System](https://github.com/user-attachments/assets/7a22f1df-f061-47b2-a357-c8ffaac61e1c)

## Project Overview

This project is part of the **NCSU GEARS Summer Research Program**, focusing on applying **Deep Reinforcement Learning (PPO algorithm)** to solve the classic Cart-Pole balance problem. The goal is to train an agent that can keep the pendulum stable on a moving cart under various initial conditions.

### Research Objectives

1. **Stability Control**: Train a PPO agent to balance the pole indefinitely without toppling
2. **Generalization**: Expand the acceptable range of initial states to improve model robustness
3. **Real-world Application**: Export trained model parameters for deployment on physical hardware (MATLAB simulation)

---

## Project Structure

Pendulum_Stability/ │ ├── myCartpoleF.py # Custom Cart-Pole environment (Gymnasium compatible) ├── Train_Result.py # PPO model training script with early stopping ├── Test_Result.py # Large-scale testing under different initial states ├── Validation.py # Statistical validation between old and new models ├── Turn-into-txt.ipynb # Export model parameters to text files (MATLAB compatible) │ ├── models/ # Saved best trained models (.zip) ├── Training/ # Training logs and intermediate model checkpoints │ ├── Logs/ # TensorBoard logs │ └── Saved Models/ # Intermediate PPO model files └── Text_Results/ # Exported model parameters (W0, b0, W1, b1, W_out, b_out)

code

---

## Environment Configuration

### Custom Cart-Pole Physics (`myCartpoleF.py`)

This project uses a **customized Cart-Pole environment** based on real-world hardware parameters from the NCSU lab setup.

#### Key Physics Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gravity` | 9.81 m/s² | Gravitational acceleration |
| `masscart` | 0.94 kg | Cart mass (0.57 + 0.37) |
| `masspole` | 0.230 kg | Pendulum mass |
| `length` | 0.3302 m | Half pole length |
| `r_mp` | 6.35×10⁻³ m | Motor pinion radius |
| `Kg` | 3.71 | Gearbox ratio |
| `Rm` | 2.6 Ω | Motor armature resistance |
| `Kt` | 0.00767 N·m/A | Motor torque constant |
| `Km` | 0.00767 V·s/rad | Back-EMF constant |
| `Beq` | 5.4 N·m·s/rad | Equivalent viscous damping |
| `Bp` | 0.0024 N·m·s/rad | Pendulum axis damping |

#### Action Space
- **Type**: `Box(-1, 1)`
- **Meaning**: Normalized voltage applied to the motor `[-10V, 10V]`

#### Observation Space
- **Type**: `Box(4)`
- **State vector**: `[x, x_dot, θ, θ_dot]`
  - `x`: Cart position (m)
  - `x_dot`: Cart velocity (m/s)
  - `θ`: Pole angle (rad)
  - `θ_dot`: Pole angular velocity (rad/s)

#### Episode Termination
- `|θ| > 0.2 rad` (~11.5°)
- `|x| > 0.2 m`
- Episode length > 2000 steps (20 seconds at 100 Hz)

#### Integrator
- Uses **Runge-Kutta 4th Order (RK4)** for higher accuracy in physics simulation

---

## Getting Started

### Prerequisites

```bash
pip install gymnasium>=0.28.0
pip install stable-baselines3>=2.0.0
pip install numpy matplotlib statsmodels
pip install pygame  # For rendering
Quick Start
1. Train a PPO Model
bash
python Train_Result.py
Training Features:

Uses PPO (Proximal Policy Optimization) with MLP policy
Early stopping when all state variables stabilize (custom StableWindow callback)
Automatically saves the best model to models/best_model.zip
Hyperparameters:
n_steps = 4096
batch_size = 256
n_epochs = 10
learning_rate = 3e-3 (linear decay)
clip_range = 0.15
2. Test Model Generalization
bash
python Test_Result.py
Testing Features:

Runs 10,000 test episodes for each initial state range
Dynamically adjusts the environment's initial state distribution
Exports results to Episode_State/episode_results_range_X.csv
Success criteria: Balance rate ≥ 40% AND episode length ≥ 1000 steps
3. Validate Model Improvement (Statistical Testing)
bash
python Validation.py
Validation Features:

Compares old_model vs new_model with 100,000 test episodes each
Performs Z-test for two proportions (one-tailed, α = 0.05)
Calculates 95% Wilson confidence intervals
Outputs statistical recommendation on whether to continue training
4. Export Model to MATLAB
bash
jupyter notebook Turn-into-txt.ipynb
Exported Files (in Text_Results/):

W0.txt, b0.txt - Layer 1 weights and biases
W1.txt, b1.txt - Layer 2 weights and biases
W_out.txt, b_out.txt - Output layer weights and biases
Results & Performance
Training Curve
The model typically achieves stable balancing after ~500,000 timesteps.

Generalization Test Results
Initial Range (±)	Success Rate	Notes
0.08	~95%	Easy initialization
0.12	~85%	Moderate difficulty
0.16	~70%	Challenging
0.20	~50%	Near failure boundary
Statistical Validation
The Validation.py script uses proportions_ztest to determine if a new model significantly outperforms the old one (p < 0.05).

Key Technical Details
PPO Network Architecture
code
Input (4) → [64, Tanh] → [64, Tanh] → Output (1)
                   ↓
            Value Function (shared layers)
Actor network: Outputs continuous action [-1, 1]
Critic network: Outputs state value estimate
Reward Design
+1 for each step the pole remains upright
Episode terminates when the pole falls or cart goes out of bounds
Custom Callbacks (Train_Result.py)
StableWindow: Monitors if a state variable stabilizes within a tolerance window
AndCallback: Stops training when all 4 state variables are stable
EvalCallback: Saves the best model based on evaluation performance
Directory Details
models/
Contains the final trained PPO model (best_model.zip) ready for deployment.

Training/Logs/
TensorBoard logs for monitoring training progress:

bash
tensorboard --logdir=Training/Logs/
Training/Saved Models/
Intermediate model checkpoints saved during training.

Text_Results/
MATLAB-compatible model parameters for hardware deployment and simulation.

Research Context
This project is developed as part of the GEARS (Global Engineering Academic Research Scholarship) program at North Carolina State University (NCSU).
