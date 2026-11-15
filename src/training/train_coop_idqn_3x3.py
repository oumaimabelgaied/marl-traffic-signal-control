
import os
import sys
import csv
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.agents.dqn_agent import DQNAgent
from src.simulation.coop_environment_3x3 import CoopMARLEnvironment3x3


SEED = 42
NUM_EPISODES = 100
NUM_SECONDS = 3600

AGENT_IDS = [
    "A0", "A1", "A2",
    "B0", "B1", "B2",
    "C0", "C1", "C2"
]

RESULTS_DIR = "results/coop_idqn_3x3_training_100ep"
MODEL_DIR = os.path.join(RESULTS_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():

    env = CoopMARLEnvironment3x3(
        num_seconds=NUM_SECONDS,
        sumo_seed=SEED,
        delta_time=15,
        use_gui=False
    )

    metrics_path = os.path.join(
        RESULTS_DIR, "training_metrics.csv"
    )

    agents = None

    try:
        with open(metrics_path, "w", newline="") as csvfile:

            fieldnames = [
                "episode",
                "agent",
                "episode_return",
                "mean_cooperative_reward",
                "average_network_queue",
                "maximum_network_queue",
                "mean_loss",
                "epsilon"
            ]

            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames
            )
            writer.writeheader()

            for episode in range(1, NUM_EPISODES + 1):

                observations = env.reset(
                    seed=SEED + episode
                )

                if agents is None:
                    agents = {
                        agent_id: DQNAgent(
                            state_dim=observations[agent_id].shape[0],
                            action_dim=2,
                            learning_rate=0.001,
                            gamma=0.99,
                            epsilon=1.0,
                            epsilon_min=0.05,
                            epsilon_decay=0.995,
                            batch_size=64,
                            memory_capacity=10000,
                            target_update=100
                        )
                        for agent_id in AGENT_IDS
                    }

                episode_returns = {
                    agent_id: 0.0
                    for agent_id in AGENT_IDS
                }

                losses = {
                    agent_id: []
                    for agent_id in AGENT_IDS
                }

                done = False

                while not done:

                    actions = {
                        agent_id: agents[agent_id].select_action(
                            observations[agent_id],
                            training=True
                        )
                        for agent_id in AGENT_IDS
                    }

                    next_observations, rewards, dones, infos = (
                        env.step(actions)
                    )

                    done = dones["__all__"]

                    for agent_id in AGENT_IDS:

                        reward = rewards[agent_id]

                        next_state = next_observations.get(
                            agent_id,
                            observations[agent_id]
                        )

                        agents[agent_id].remember(
                            observations[agent_id],
                            actions[agent_id],
                            reward,
                            next_state,
                            done
                        )

                        loss = agents[agent_id].train_step()

                        if loss is not None:
                            losses[agent_id].append(loss)

                        episode_returns[agent_id] += reward

                    observations = next_observations

                queue_history = env.queue_history

                average_queue = (
                    float(np.mean(queue_history))
                    if queue_history else 0.0
                )

                maximum_queue = (
                    int(max(queue_history))
                    if queue_history else 0
                )

                for agent_id in AGENT_IDS:

                    agents[agent_id].decay_epsilon()

                    mean_reward = (
                        episode_returns[agent_id]
                        / max(1, NUM_SECONDS // 15)
                    )

                    mean_loss = (
                        float(np.mean(losses[agent_id]))
                        if losses[agent_id] else 0.0
                    )

                    writer.writerow({
                        "episode": episode,
                        "agent": agent_id,
                        "episode_return": episode_returns[agent_id],
                        "mean_cooperative_reward": mean_reward,
                        "average_network_queue": average_queue,
                        "maximum_network_queue": maximum_queue,
                        "mean_loss": mean_loss,
                        "epsilon": agents[agent_id].epsilon
                    })

                csvfile.flush()

                print(
                    f"Episode {episode}/{NUM_EPISODES} | "
                    f"Network queue: {average_queue:.3f} | "
                    f"Max queue: {maximum_queue}"
                )

                for agent_id in AGENT_IDS:
                    print(
                        f"  {agent_id}: "
                        f"Return={episode_returns[agent_id]:.2f}, "
                        f"Loss={np.mean(losses[agent_id]) if losses[agent_id] else 0:.4f}, "
                        f"Epsilon={agents[agent_id].epsilon:.3f}"
                    )

        for agent_id, agent in agents.items():

            checkpoint = {
                "policy_net": agent.policy_net.state_dict(),
                "target_net": agent.target_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "learning_steps": agent.learning_steps
            }

            torch.save(
                checkpoint,
                os.path.join(
                    MODEL_DIR,
                    f"coop_dqn_agent_{agent_id}.pth"
                )
            )

        print("\n3x3 Coop-IDQN training completed successfully.")
        print("Metrics:", metrics_path)
        print("Models:", MODEL_DIR)

    finally:
        env.close()


if __name__ == "__main__":
    main()
