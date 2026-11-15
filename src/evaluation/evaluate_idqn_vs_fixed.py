
import os
import csv
import xml.etree.ElementTree as ET

import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.simulation.marl_environment_2x2 import MARLEnvironment2x2


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SEEDS = list(range(143, 153))
NUM_SECONDS = 3600
DELTA_TIME = 15

AGENT_IDS = ["1", "2", "5", "6"]

MODEL_DIR = "results/idqn_2x2_training_100ep/models"
RESULTS_DIR = "results/evaluation_idqn_vs_fixed"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINED AGENTS
# ============================================================

def load_agents():

    agents = {}

    for agent_id in AGENT_IDS:

        agent = DQNAgent(
            state_dim=21,
            action_dim=4
        )

        model_path = os.path.join(
            MODEL_DIR,
            f"dqn_agent_{agent_id}.pth"
        )

        checkpoint = torch.load(
            model_path,
            map_location=agent.device,
            weights_only=False
        )

        agent.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        agent.policy_net.eval()

        agents[agent_id] = agent

        print(f"Agent {agent_id}: model loaded")

    return agents


# ============================================================
# READ TRIPINFO XML
# ============================================================

def read_tripinfo(filepath):

    root = ET.parse(filepath).getroot()
    trips = root.findall("tripinfo")

    if not trips:
        return {
            "arrived_vehicles": 0,
            "average_waiting_time": 0.0,
            "average_travel_time": 0.0,
            "average_time_loss": 0.0,
        }

    return {
        "arrived_vehicles": len(trips),

        "average_waiting_time": np.mean([
            float(t.get("waitingTime"))
            for t in trips
        ]),

        "average_travel_time": np.mean([
            float(t.get("duration"))
            for t in trips
        ]),

        "average_time_loss": np.mean([
            float(t.get("timeLoss"))
            for t in trips
        ]),
    }


# ============================================================
# EVALUATE ONE SIMULATION
# ============================================================

def evaluate_run(method, seed, agents=None):

    fixed_ts = method == "Fixed-time"

    tripinfo_path = os.path.join(
        RESULTS_DIR,
        f"tripinfo_{method.lower().replace('-', '_')}_seed{seed}.xml"
    )

    env = MARLEnvironment2x2(
        delta_time=DELTA_TIME,
        yellow_time=3,
        min_green=10,
        max_green=50,
        num_seconds=NUM_SECONDS,
        sumo_seed=seed,
        use_gui=False,
        fixed_ts=fixed_ts,
        additional_sumo_cmd=f"--tripinfo-output {tripinfo_path}",
    )

    try:

        observations = env.reset(seed=seed)

        done = False
        decision_count = 0

        while not done:

            if method == "IDQN":

                actions = {
                    agent_id: agents[agent_id].select_action(
                        observations[agent_id],
                        training=False
                    )
                    for agent_id in AGENT_IDS
                }

            else:

                # The native SUMO traffic-light program
                # controls the signals in fixed-time mode.
                actions = {
                    agent_id: 0
                    for agent_id in AGENT_IDS
                }

            observations, rewards, dones, infos = env.step(actions)

            done = dones["__all__"]
            decision_count += 1

        queue_history = env.queue_history.copy()
        simulation_time = env.sim_step

    finally:
        env.close()

    trip_metrics = read_tripinfo(tripinfo_path)

    results = {
        "method": method,
        "seed": seed,
        "simulation_time": simulation_time,
        "decisions": decision_count,
        "queue_samples": len(queue_history),
        "average_queue": float(np.mean(queue_history))
            if queue_history else 0.0,
        "maximum_queue": int(max(queue_history))
            if queue_history else 0,
        **trip_metrics,
    }

    return results


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n========== IDQN vs FIXED-TIME ==========")
    print(f"Test seeds: {TEST_SEEDS}")
    print(f"Simulation duration: {NUM_SECONDS} seconds")

    agents = load_agents()
    all_results = []

    for seed in TEST_SEEDS:

        print(f"\n\n========== SEED {seed} ==========")

        for method in ["IDQN", "Fixed-time"]:

            print(f"\n========== {method} ==========")

            result = evaluate_run(
                method=method,
                seed=seed,
                agents=agents if method == "IDQN" else None
            )

            all_results.append(result)

            print(
                f"Queue: {result['average_queue']:.3f} | "
                f"Waiting: {result['average_waiting_time']:.3f} | "
                f"Travel: {result['average_travel_time']:.3f} | "
                f"Arrived: {result['arrived_vehicles']}"
            )

    # Save raw results
    csv_path = os.path.join(
        RESULTS_DIR,
        "comparison_all_seeds.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Calculate mean and population standard deviation
    metrics = [
        "average_queue",
        "average_waiting_time",
        "average_travel_time",
        "average_time_loss",
        "arrived_vehicles",
    ]

    print("\n\n========== FINAL STATISTICS ==========")

    for method in ["IDQN", "Fixed-time"]:

        method_results = [
            result for result in all_results
            if result["method"] == method
        ]

        print(f"\n{method}")

        for metric in metrics:

            values = [
                result[metric]
                for result in method_results
            ]

            mean = float(np.mean(values))
            std = float(np.std(values))

            print(
                f"{metric}: {mean:.3f} ± {std:.3f}"
            )

    print(f"\nRaw results saved to: {csv_path}")


if __name__ == "__main__":
    main()
