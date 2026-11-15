
import csv
import random
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import traci
from sumolib import checkBinary

from marl_environment import TrafficSignalEnvironment
from dqn_agent import DQNAgent


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

NETWORK_FILE = PROJECT_DIR / "SUMO" / "networks" / "test3.net.xml"
ROUTE_FILE = PROJECT_DIR / "SUMO" / "routes" / "test3.rou.xml"

MODEL_DIR = PROJECT_DIR / "results" / "models"
RESULTS_DIR = PROJECT_DIR / "results"

AGENTS = ["A1", "B0", "B1", "B2", "C1"]

DECISION_INTERVAL = 15
WARMUP_STEPS = 20

MAX_DECISIONS = 200
MAX_SIMULATION_STEPS = 5000

EVALUATION_SCENARIOS = [
    {
        "route_file": PROJECT_DIR / "SUMO" / "routes" / "evaluation" / f"eval_seed_{seed}.rou.xml",
        "seed": seed
    }
    for seed in range(101, 106)
]

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_incoming_lanes(tls_id):

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    return list(dict.fromkeys(
        link[0]
        for signal_group in controlled_links
        for link in signal_group
        if link
    ))


def get_total_queue(incoming_lanes):

    return sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in incoming_lanes
    )


def calculate_average_queue(queue_area, elapsed_steps):

    if elapsed_steps == 0:
        return 0.0

    return queue_area / (len(AGENTS) * elapsed_steps)


def read_tripinfo(tripinfo_file):

    if not tripinfo_file.exists():
        raise FileNotFoundError(
            f"Trip information file not found: {tripinfo_file}"
        )

    total_waiting_time = 0.0
    total_travel_time = 0.0
    total_time_loss = 0.0
    completed_trips = 0

    for _, element in ET.iterparse(
        tripinfo_file,
        events=("end",)
    ):

        if element.tag == "tripinfo":

            total_waiting_time += float(
                element.attrib["waitingTime"]
            )

            total_travel_time += float(
                element.attrib["duration"]
            )

            total_time_loss += float(
                element.attrib.get("timeLoss", 0.0)
            )

            completed_trips += 1
            element.clear()

    if completed_trips == 0:

        return {
            "average_waiting_time": 0.0,
            "average_travel_time": 0.0,
            "average_time_loss": 0.0,
            "completed_trips": 0
        }

    return {
        "average_waiting_time": total_waiting_time / completed_trips,
        "average_travel_time": total_travel_time / completed_trips,
        "average_time_loss": total_time_loss / completed_trips,
        "completed_trips": completed_trips
    }


# ==================================================
# 1. FIXED-TIME EVALUATION
# ==================================================

def evaluate_fixed_time(route_file, seed):

    print(f"\n  Fixed-time | Seed {seed}")

    tripinfo_file = RESULTS_DIR / f"tripinfo_fixed_{seed}.xml"
    tripinfo_file.unlink(missing_ok=True)

    sumo_cmd = [
        checkBinary("sumo"),
        "-n", str(NETWORK_FILE),
        "-r", str(route_file),
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--seed", str(seed),
        "--tripinfo-output", str(tripinfo_file)
    ]

    total_queue_area = 0.0
    elapsed_steps = 0
    arrived_vehicles = 0

    try:

        traci.start(sumo_cmd)

        # Warm-up
        for _ in range(WARMUP_STEPS):

            if traci.simulation.getMinExpectedNumber() == 0:
                break

            traci.simulationStep()

        incoming_lanes = {
            tls_id: get_incoming_lanes(tls_id)
            for tls_id in AGENTS
        }

        while (
            traci.simulation.getMinExpectedNumber() > 0
            and elapsed_steps < MAX_SIMULATION_STEPS
        ):

            traci.simulationStep()
            elapsed_steps += 1

            arrived_vehicles += (
                traci.simulation.getArrivedNumber()
            )

            total_queue_area += sum(
                get_total_queue(lanes)
                for lanes in incoming_lanes.values()
            )

    finally:

        traci.close()

    trip_metrics = read_tripinfo(tripinfo_file)

    return {
        "seed": seed,
        "method": "Fixed-time",
        "average_queue": calculate_average_queue(
            total_queue_area, elapsed_steps
        ),
        "average_waiting_time": trip_metrics["average_waiting_time"],
        "average_travel_time": trip_metrics["average_travel_time"],
        "average_time_loss": trip_metrics["average_time_loss"],
        "arrived_vehicles": arrived_vehicles,
        "completed_trips": trip_metrics["completed_trips"],
        "simulation_steps": elapsed_steps
    }


# ==================================================
# 2. RANDOM AND IDQN EVALUATION
# ==================================================

def evaluate_rl(method, route_file, seed):

    print(f"\n  {method.upper()} | Seed {seed}")

    tripinfo_file = RESULTS_DIR / f"tripinfo_{method}_{seed}.xml"
    tripinfo_file.unlink(missing_ok=True)

    env = TrafficSignalEnvironment(
        decision_interval=DECISION_INTERVAL,
        warmup_steps=WARMUP_STEPS,
        route_file=route_file,
        seed=seed
    )


    # Add the tripinfo output file
    env.sumo_cmd.extend([
        "--tripinfo-output", str(tripinfo_file)
    ])

    agents = {}

    # Load trained DQN models only once per evaluation
    if method == "idqn":

        for tls_id in AGENTS:

            agent = DQNAgent()

            model_path = MODEL_DIR / f"{tls_id}_dqn.pth"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )

            state_dict = torch.load(
                model_path,
                map_location=agent.device,
                weights_only=True
            )

            agent.policy_net.load_state_dict(state_dict)
            agent.policy_net.eval()

            agent.epsilon = 0.0

            agents[tls_id] = agent

    total_queue_area = 0.0
    total_elapsed_steps = 0

    arrived_vehicles = 0
    decisions = 0

    try:

        observations = env.reset()

        while decisions < MAX_DECISIONS:

            actions = {}

            for tls_id in AGENTS:

                if method == "random":

                    actions[tls_id] = random.randint(0, 1)

                elif method == "idqn":

                    state = observations[tls_id]["normalized_state"]

                    actions[tls_id] = agents[tls_id].select_action(
                        state,
                        training=False
                    )

            observations, rewards, info = env.step(actions)

            total_queue_area += sum(
                info["queue_area"].values()
            )

            total_elapsed_steps += info["elapsed_steps"]

            arrived_vehicles += info["arrived_vehicles"]

            decisions += 1

            if traci.simulation.getMinExpectedNumber() == 0:
                break

    finally:

        env.close()

    trip_metrics = read_tripinfo(tripinfo_file)

    return {
        "seed": seed,
        "method": method.upper(),
        "average_queue": calculate_average_queue(
            total_queue_area, total_elapsed_steps
        ),
        "average_waiting_time": trip_metrics["average_waiting_time"],
        "average_travel_time": trip_metrics["average_travel_time"],
        "average_time_loss": trip_metrics["average_time_loss"],
        "arrived_vehicles": arrived_vehicles,
        "completed_trips": trip_metrics["completed_trips"],
        "simulation_steps": total_elapsed_steps
    }


# ==================================================
# 3. MULTI-SEED EXPERIMENT
# ==================================================


def run_experiments():

    all_results = []

    for scenario in EVALUATION_SCENARIOS:

        route_file = scenario["route_file"]
        seed = scenario["seed"]

        print("\n" + "=" * 65)
        print(f"UNSEEN ROUTE SCENARIO: {seed}")
        print(f"Route file: {route_file.name}")
        print("=" * 65)

        if not route_file.exists():
            raise FileNotFoundError(
                f"Route file not found: {route_file}"
            )

        all_results.append(
            evaluate_fixed_time(route_file, seed)
        )

        random.seed(seed)

        all_results.append(
            evaluate_rl("random", route_file, seed)
        )

        all_results.append(
            evaluate_rl("idqn", route_file, seed)
        )

    return all_results



# ==================================================
# 4. CALCULATE MEAN AND STANDARD DEVIATION
# ==================================================

def calculate_summary(results):

    metrics = [
        "average_queue",
        "average_waiting_time",
        "average_travel_time",
        "average_time_loss",
        "arrived_vehicles"
    ]

    summary = []

    methods = ["Fixed-time", "RANDOM", "IDQN"]

    for method in methods:

        method_results = [
            result for result in results
            if result["method"] == method
        ]

        row = {
            "method": method,
            "runs": len(method_results)
        }

        for metric in metrics:

            values = [
                result[metric]
                for result in method_results
            ]

            row[f"{metric}_mean"] = statistics.mean(values)

            row[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1 else 0.0
            )

        summary.append(row)

    return summary


# ==================================================
# 5. SAVE CSV FILES
# ==================================================

def save_results(all_results, summary):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Individual run results
    raw_file = RESULTS_DIR / "unseen_routes_results.csv"

    with open(raw_file, "w", newline="") as csvfile:

        fieldnames = list(all_results[0].keys())

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Mean and standard deviation
    summary_file = RESULTS_DIR / "unseen_routes_summary.csv"

    with open(summary_file, "w", newline="") as csvfile:

        fieldnames = list(summary[0].keys())

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary)

    print(f"\nIndividual results saved to: {raw_file}")
    print(f"Summary saved to: {summary_file}")


# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

def display_summary(summary):

    print("\n")
    print("=" * 90)
    print("UNSEEN ROUTE GENERALIZATION SUMMARY")
    print("=" * 90)

    for row in summary:

        print(f"\nMethod: {row['method']}")
        print(f"Number of runs: {row['runs']}")

        print(
            f"Average queue: "
            f"{row['average_queue_mean']:.3f} "
            f"+/- {row['average_queue_std']:.3f}"
        )

        print(
            f"Average waiting time: "
            f"{row['average_waiting_time_mean']:.2f} "
            f"+/- {row['average_waiting_time_std']:.2f} seconds"
        )

        print(
            f"Average travel time: "
            f"{row['average_travel_time_mean']:.2f} "
            f"+/- {row['average_travel_time_std']:.2f} seconds"
        )

        print(
            f"Average time loss: "
            f"{row['average_time_loss_mean']:.2f} "
            f"+/- {row['average_time_loss_std']:.2f} seconds"
        )

        print(
            f"Average arrived vehicles: "
            f"{row['arrived_vehicles_mean']:.1f} "
            f"+/- {row['arrived_vehicles_std']:.2f}"
        )


# ==================================================
# MAIN
# ==================================================


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("UNSEEN TRAFFIC ROUTE GENERALIZATION EVALUATION")
    print(f"Scenarios: {len(EVALUATION_SCENARIOS)}")
    print(f"Controllers: Fixed-time, Random, IDQN")
    print(f"Total simulations: {len(EVALUATION_SCENARIOS) * 3}")
    print("=" * 65)

    all_results = run_experiments()

    summary = calculate_summary(all_results)

    display_summary(summary)

    save_results(all_results, summary)



if __name__ == "__main__":

    main()
