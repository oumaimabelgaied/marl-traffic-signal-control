
import os
import sys
import csv
import xml.etree.ElementTree as ET

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.agents.dqn_agent import DQNAgent
from src.simulation.marl_environment_3x3 import MARLEnvironment3x3
from src.simulation.coop_environment_3x3 import CoopMARLEnvironment3x3


TEST_SEEDS = list(range(153, 163))
NUM_SECONDS = 3600
DELTA_TIME = 15

AGENT_IDS = [
    "A0", "A1", "A2",
    "B0", "B1", "B2",
    "C0", "C1", "C2"
]

IDQN_MODEL_DIR = "results/idqn_3x3_training_100ep/models"
COOP_MODEL_DIR = "results/coop_idqn_3x3_training_100ep/models"

RESULTS_DIR = "results/evaluation_all_methods_3x3"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_agents(model_dir, filename_prefix):

    agents = {}

    for agent_id in AGENT_IDS:

        agent = DQNAgent(
            state_dim=11,
            action_dim=2
        )

        checkpoint_path = os.path.join(
            model_dir,
            f"{filename_prefix}{agent_id}.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=agent.device,
            weights_only=False
        )

        agent.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        agent.policy_net.eval()
        agents[agent_id] = agent

    return agents


def read_tripinfo(path):

    if not os.path.exists(path):
        return {
            "arrived_vehicles": 0,
            "average_waiting_time": 0.0,
            "average_travel_time": 0.0,
            "average_time_loss": 0.0
        }

    root = ET.parse(path).getroot()
    trips = root.findall("tripinfo")

    if not trips:
        return {
            "arrived_vehicles": 0,
            "average_waiting_time": 0.0,
            "average_travel_time": 0.0,
            "average_time_loss": 0.0
        }

    return {
        "arrived_vehicles": len(trips),
        "average_waiting_time": float(np.mean([
            float(t.get("waitingTime", 0))
            for t in trips
        ])),
        "average_travel_time": float(np.mean([
            float(t.get("duration", 0))
            for t in trips
        ])),
        "average_time_loss": float(np.mean([
            float(t.get("timeLoss", 0))
            for t in trips
        ]))
    }


def evaluate_run(method, seed, idqn_agents=None, coop_agents=None):

    fixed_ts = method == "Fixed-time"

    tripinfo_path = os.path.join(
        RESULTS_DIR,
        f"tripinfo_{method.lower().replace('-', '_')}_seed{seed}.xml"
    )

    if method == "Coop-IDQN":
        env = CoopMARLEnvironment3x3(
            num_seconds=NUM_SECONDS,
            sumo_seed=seed,
            delta_time=DELTA_TIME,
            use_gui=False,
            additional_sumo_cmd=f"--tripinfo-output {tripinfo_path}"
        )
        active_agents = coop_agents

    else:
        env = MARLEnvironment3x3(
            num_seconds=NUM_SECONDS,
            sumo_seed=seed,
            delta_time=DELTA_TIME,
            use_gui=False,
            fixed_ts=fixed_ts,
            additional_sumo_cmd=f"--tripinfo-output {tripinfo_path}"
        )
        active_agents = idqn_agents

    try:
        observations = env.reset(seed=seed)

        decisions = 0
        done = False

        while not done:

            if fixed_ts:
                actions = {
                    agent_id: 0 for agent_id in AGENT_IDS
                }

            else:
                actions = {
                    agent_id: active_agents[agent_id].select_action(
                        observations[agent_id],
                        training=False
                    )
                    for agent_id in AGENT_IDS
                }

            next_observations, rewards, dones, infos = env.step(actions)

            done = dones["__all__"]
            decisions += 1
            observations = next_observations

        queue_history = env.queue_history

        queue_metrics = {
            "average_queue": float(np.mean(queue_history)),
            "maximum_queue": int(max(queue_history)) if queue_history else 0
        }

        simulation_time = env.sim_step

    finally:
        env.close()

    trip_metrics = read_tripinfo(tripinfo_path)

    return {
        "method": method,
        "seed": seed,
        "simulation_time": simulation_time,
        "decisions": decisions,
        "queue_samples": len(queue_history),
        **queue_metrics,
        **trip_metrics
    }


def main():

    print("Loading IDQN models...")
    idqn_agents = load_agents(
        IDQN_MODEL_DIR,
        "dqn_agent_"
    )

    print("Loading Coop-IDQN models...")
    coop_agents = load_agents(
        COOP_MODEL_DIR,
        "coop_dqn_agent_"
    )

    results = []

    methods = [
        "IDQN",
        "Coop-IDQN",
        "Fixed-time"
    ]

    for seed in TEST_SEEDS:

        print(f"\n========== SEED {seed} ==========")

        for method in methods:

            print(f"Evaluating {method}...")

            result = evaluate_run(
                method,
                seed,
                idqn_agents=idqn_agents,
                coop_agents=coop_agents
            )

            results.append(result)

            print(
                f"Queue={result['average_queue']:.3f} | "
                f"Waiting={result['average_waiting_time']:.3f} | "
                f"Travel={result['average_travel_time']:.3f} | "
                f"Time loss={result['average_time_loss']:.3f} | "
                f"Arrived={result['arrived_vehicles']}"
            )

    csv_path = os.path.join(
        RESULTS_DIR,
        "comparison_all_seeds.csv"
    )

    with open(csv_path, "w", newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== FINAL STATISTICS ==========")

    for method in methods:

        subset = [
            row for row in results
            if row["method"] == method
        ]

        print(f"\n{method}")

        for metric in [
            "average_queue",
            "average_waiting_time",
            "average_travel_time",
            "average_time_loss",
            "arrived_vehicles"
        ]:

            values = [
                row[metric] for row in subset
            ]

            print(
                f"{metric}: "
                f"{np.mean(values):.3f} ± {np.std(values):.3f}"
            )

    print("\nResults saved to:", csv_path)


if __name__ == "__main__":
    main()
