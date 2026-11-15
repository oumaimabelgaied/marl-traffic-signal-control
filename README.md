# Multi-Agent Reinforcement Learning for Traffic Signal Control

**INSAT — Networks and Telecommunications Engineering (RT5)**  
**Project:** Multi-agent reinforcement learning for adaptive traffic signal control using SUMO.

> This repository contains the implementation and experiments conducted for the academic project. Numerical results should be interpreted only with the configurations, seeds, and evaluation protocols documented in the source files and experiment outputs.

## 1. Project overview

The project studies decentralized reinforcement-learning approaches for urban traffic-signal control. Independent DQN agents (IDQN) and cooperative DQN agents (Coop-IDQN) are evaluated against SUMO's native fixed-time traffic-light programs.

Two grid topologies are considered:

- **2×2 grid:** four controlled intersections (`1`, `2`, `5`, `6`), with 21-dimensional observations and four available green-phase actions per agent.
- **3×3 grid:** nine RL-controlled intersections (`A0`–`C2`), with 11-dimensional observations and two available green-phase actions per agent. Peripheral signals retain their native SUMO programs.

An additional CO₂-emissions evaluation is implemented for the 2×2 topology using SUMO's vehicle-emission estimates.

## 2. Methods

- **Fixed-time:** native traffic-light programs defined in the SUMO network.
- **IDQN:** one independent DQN agent per controlled intersection.
- **Coop-IDQN:** independent DQN learners trained with a cooperative reward that combines local and neighboring agents' rewards.
- **Advanced extensions:** V2X latency/packet-loss and federated-learning experiments are being documented separately from the core experiments. Their scripts, configurations, data provenance, and authorship must be identified explicitly before their results are merged into the paper.

## 3. Repository structure

The following is the recommended organization. Adapt names to the files actually present in the repository; do not move files until imports and paths have been checked.

```text
MARL/
├── README.md
├── .gitignore
├── requirements.txt                 # or environment.yml
├── SUMO/
│   ├── networks/
│   │   ├── 2x2grid/
│   │   └── 3x3grid/
│   └── routes/
│       ├── 2x2grid/
│       └── 3x3grid/
├── src/
│   ├── agents/
│   ├── simulation/
│   ├── training/
│   └── evaluation/
├── results/
│   ├── idqn_2x2_training_100ep/
│   ├── coop_idqn_2x2_training_100ep/
│   ├── idqn_3x3_training_100ep/
│   ├── coop_idqn_3x3_training_100ep/
│   ├── evaluation_all_methods/
│   ├── evaluation_all_methods_co2/
│   └── evaluation_all_methods_3x3/
├── docs/
│   └── paper/
└── scripts/
```

Large generated outputs (checkpoints, raw XML, plots) may be kept locally or stored with Git LFS / an external archive if repository size becomes excessive. Keep the scripts and a small manifest describing where omitted artifacts can be obtained.

## 4. Experimental protocol currently used

| Experiment | Training episodes | Training seeds | Evaluation seeds | Simulation horizon |
|---|---:|---|---|---:|
| 2×2 IDQN / Coop-IDQN | 100 | 43–142 | 143–152 | 3,600 s |
| 3×3 IDQN / Coop-IDQN | 100 | 43–142 | 153–162 | 3,600 s |
| 2×2 CO₂ evaluation | Existing trained policies | — | 143–152 | 3,600 s |

The DQN agents are evaluated greedily (`training=False`). Reported summary values use the mean and population standard deviation across ten evaluation seeds. The detailed seed-level CSV files should be retained to support paired comparisons and statistical analysis.

## 5. Metrics

- Average queue length (as implemented by the project environment).
- Average waiting time of completed trips.
- Average travel time of completed trips.
- Average time loss of completed trips.
- Number of arrived vehicles.
- Estimated total CO₂ emissions (2×2 evaluation).
- Estimated CO₂ emissions per arrived vehicle (2×2 evaluation).

**Important:** SUMO's CO₂ values are simulation-model estimates, not physical measurements. Total emissions are accumulated over active vehicles at each simulation step, whereas trip-based metrics are computed from completed trips. Vehicles still in the network at the end of an episode therefore contribute to total emissions but are not counted as arrived vehicles.

## 6. Setup

Requirements include a compatible SUMO installation, SUMO-RL, TraCI, NumPy, pandas, PyTorch, Gymnasium, PettingZoo, and Matplotlib. The exact installed versions should be recorded in `requirements.txt` or an environment export before final publication.

On Windows PowerShell, from the repository root:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Verify SUMO is available
sumo --version

# Compile-check a Python module
python -m py_compile src\simulation\marl_environment_2x2.py
```

Set `SUMO_HOME` to the installed SUMO directory if it is not already configured. Use the project's existing virtual environment rather than installing packages globally.

## 7. Running experiments

Run scripts from the repository root so relative paths to `SUMO/` and `results/` resolve correctly. Examples:

```powershell
# 2×2 training / evaluation
python -m src.training.train_idqn_2x2
python -m src.evaluation.evaluate_all_methods

# 3×3 training / evaluation
python -m src.training.train_idqn_3x3
python -m src.evaluation.evaluate_all_methods_3x3
```

Check the actual filenames in `src/training/` and `src/evaluation/` before running. Training commands should not be rerun merely to reproduce evaluation: evaluation scripts load the saved checkpoints.

## 8. Reproducibility and reporting

For every reported experiment, record:

1. SUMO and Python/package versions.
2. Network and route files, demand-generation procedure, and simulation horizon.
3. Training and evaluation seeds.
4. Observation/action dimensions, reward definition, DQN hyperparameters, and cooperation coefficient.
5. Checkpoint identifiers and evaluation script version.
6. Raw seed-level outputs and the exact aggregation method.

Do not combine results from different network files, traffic demands, horizons, or training protocols without clearly identifying the differences. External or teammate-produced experiments must be attributed and described with their own methodology; do not present them as having been run by this repository unless that is accurate.

## 9. Current known results (descriptive, pending seed-level analysis)

The following are the current ten-seed summary values reported during development. They are included as a working snapshot and should be regenerated from the archived CSV files before final submission.

### 2×2 network

| Method | Mean queue | Waiting time (s) | Travel time (s) | Time loss (s) | Arrived vehicles | CO₂ total (kg) | CO₂ / arrived vehicle (kg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDQN | 2.072 | 5.204 | 46.376 | 13.512 | 1424.4 | 150.825 | 0.105885 |
| Coop-IDQN | 3.643 | 9.125 | 51.294 | 18.428 | 1420.9 | 163.886 | 0.115348 |
| Fixed-time | 10.983 | 27.579 | 70.620 | 37.754 | 1416.0 | 202.277 | 0.142838 |

### 3×3 network

| Method | Mean queue | Waiting time (s) | Travel time (s) | Time loss (s) | Arrived vehicles |
|---|---:|---:|---:|---:|---:|
| IDQN | 2.824 | 9.926 | 80.568 | 23.137 | 1048.4 |
| Coop-IDQN | 2.910 | 10.276 | 81.040 | 23.609 | 1047.7 |
| Fixed-time | 11.779 | 40.928 | 120.502 | 63.134 | 1030.9 |

These are descriptive summaries, not a substitute for paired seed-level statistical tests or uncertainty analysis. Avoid interpreting small differences as statistically significant without those analyses.

## 10. License, attribution, and academic integrity

Add the license required by the course or institution before public release. Cite SUMO, SUMO-RL, relevant reinforcement-learning literature, and any external datasets or contributed experiments. Clearly distinguish original implementation, adapted third-party code, and teammate-provided results.
