
import os
import csv
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.agents.dqn_agent import DQNAgent
from src.simulation.marl_environment_2x2 import MARLEnvironment2x2
from src.simulation.coop_environment_2x2 import (
    MARLEnvironment2x2 as CoopMARLEnvironment2x2
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SEEDS = list(range(143, 153))
NUM_SECONDS = 3600
DELTA_TIME = 15

AGENT_IDS = ["1", "2", "5", "6"]

IDQN_MODEL_DIR = "results/idqn_2x2_training_100ep/models"
COOP_MODEL_DIR = "results/coop_idqn_2x2_training_100ep/models"

# Separate folder to preserve previous evaluation results.
RESULTS_DIR = "results/evaluation_all_methods_co2"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINED AGENTS
# ============================================================

def load_agents(model_dir, filename_prefix):

    agents = {}

    for agent_id in AGENT_IDS:

        agent = DQNAgent(
            state_dim=21,
            action_dim=4
        )

        model_path = os.path.join(
            model_dir,
            f"{filename_prefix}{agent_id}.pth"
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

        "average_waiting_time": float(np.mean([
            float(t.get("waitingTime"))
            for t in trips
        ])),

        "average_travel_time": float(np.mean([
            float(t.get("duration"))
            for t in trips
        ])),

        "average_time_loss": float(np.mean([
            float(t.get("timeLoss"))
            for t in trips
        ])),
    }


# ============================================================
# EVALUATE ONE SIMULATION
# ============================================================

def evaluate_run(method, seed, agents=None):

    if method == "Fixed-time":
        env_class = MARLEnvironment2x2
        fixed_ts = True

    elif method == "IDQN":
        env_class = MARLEnvironment2x2
        fixed_ts = False

    elif method == "Coop-IDQN":
        env_class = CoopMARLEnvironment2x2
        fixed_ts = False

    else:
        raise ValueError(f"Unknown method: {method}")

    tripinfo_path = os.path.join(
        RESULTS_DIR,
        f"tripinfo_{method.lower().replace('-', '_')}_seed{seed}.xml"
    )

    env = env_class(
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

            if method in ["IDQN", "Coop-IDQN"]:

                actions = {
                    agent_id: agents[agent_id].select_action(
                        observations[agent_id],
                        training=False
                    )
                    for agent_id in AGENT_IDS
                }

            else:

                # Native SUMO traffic-light program
                # controls the signals in fixed-time mode.
                actions = {
                    agent_id: 0
                    for agent_id in AGENT_IDS
                }

            observations, rewards, dones, infos = env.step(actions)

            done = dones["__all__"]
            decision_count += 1

        # Collect queue and CO2 histories before closing SUMO.
        queue_history = env.queue_history.copy()
        simulation_time = env.sim_step

        total_co2_mg = env.total_co2_mg
        total_co2_kg = env.total_co2_kg

    finally:
        env.close()

    # Read completed vehicle trips.
    trip_metrics = read_tripinfo(tripinfo_path)

    arrived_vehicles = trip_metrics["arrived_vehicles"]

    # Normalize emissions by the number of arrived vehicles.
    co2_per_arrived_vehicle = (
        total_co2_kg / arrived_vehicles
        if arrived_vehicles > 0
        else 0.0
    )

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

        "total_co2_mg": total_co2_mg,
        "total_co2_kg": total_co2_kg,
        "co2_per_arrived_vehicle_kg": co2_per_arrived_vehicle,

        **trip_metrics,
    }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    methods = ["IDQN", "Coop-IDQN", "Fixed-time"]

    print("\n========== IDQN vs COOP-IDQN vs FIXED-TIME ==========")
    print(f"Test seeds: {TEST_SEEDS}")
    print(f"Simulation duration: {NUM_SECONDS} seconds")
    print("Additional metric: CO2 emissions")

    print("\nLoading IDQN models...")
    idqn_agents = load_agents(
        IDQN_MODEL_DIR,
        "dqn_agent_"
    )

    print("\nLoading Coop-IDQN models...")
    coop_agents = load_agents(
        COOP_MODEL_DIR,
        "coop_dqn_agent_"
    )

    all_results = []

    for seed in TEST_SEEDS:

        print(f"\n\n========== SEED {seed} ==========")

        for method in methods:

            print(f"\n========== {method} ==========")

            if method == "IDQN":
                current_agents = idqn_agents

            elif method == "Coop-IDQN":
                current_agents = coop_agents

            else:
                current_agents = None

            result = evaluate_run(
                method=method,
                seed=seed,
                agents=current_agents
            )

            all_results.append(result)

            print(
                f"Queue: {result['average_queue']:.3f} | "
                f"Waiting: {result['average_waiting_time']:.3f} | "
                f"Travel: {result['average_travel_time']:.3f} | "
                f"Time loss: {result['average_time_loss']:.3f} | "
                f"Arrived: {result['arrived_vehicles']} | "
                f"CO2: {result['total_co2_kg']:.3f} kg | "
                f"CO2/vehicle: "
                f"{result['co2_per_arrived_vehicle_kg']:.6f} kg"
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    csv_path = os.path.join(
        RESULTS_DIR,
        "comparison_all_seeds_with_co2.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    metrics = [
        "average_queue",
        "average_waiting_time",
        "average_travel_time",
        "average_time_loss",
        "arrived_vehicles",
        "total_co2_kg",
        "co2_per_arrived_vehicle_kg",
    ]

    print("\n\n========== FINAL STATISTICS ==========")

    for method in methods:

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
                f"{metric}: {mean:.6f} ± {std:.6f}"
            )

    # ========================================================
    # CO2 REDUCTION VS FIXED-TIME
    # ========================================================

    fixed_results = [
        r for r in all_results
        if r["method"] == "Fixed-time"
    ]

    baseline_co2 = float(np.mean([
        r["co2_per_arrived_vehicle_kg"]
        for r in fixed_results
    ]))

    baseline_total_co2 = float(np.mean([
        r["total_co2_kg"]
        for r in fixed_results
    ]))

    print("\n========== CO2 IMPROVEMENT VS FIXED-TIME ==========")

    for method in ["IDQN", "Coop-IDQN"]:

        method_results = [
            r for r in all_results
            if r["method"] == method
        ]

        method_co2_per_vehicle = float(np.mean([
            r["co2_per_arrived_vehicle_kg"]
            for r in method_results
        ]))

        method_total_co2 = float(np.mean([
            r["total_co2_kg"]
            for r in method_results
        ]))

        reduction_per_vehicle = (
            (baseline_co2 - method_co2_per_vehicle)
            / baseline_co2 * 100
            if baseline_co2 > 0 else 0.0
        )

        reduction_total = (
            (baseline_total_co2 - method_total_co2)
            / baseline_total_co2 * 100
            if baseline_total_co2 > 0 else 0.0
        )

        print(f"\n{method}")
        print(
            f"CO2 per vehicle reduction: "
            f"{reduction_per_vehicle:.2f}%"
        )
        print(
            f"Total CO2 reduction: "
            f"{reduction_total:.2f}%"
        )

    print(f"\nRaw results saved to: {csv_path}")


if __name__ == "__main__":
    main()
