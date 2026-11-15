
import os
import csv

import numpy as np
import torch
import traci

from marl_environment import TrafficSignalEnvironment
from dqn_agent import DQNAgent


# --------------------------------------------------
# 1. Training configuration
# --------------------------------------------------

NUM_EPISODES = 10
MAX_DECISIONS = 100

AGENTS = ["A1", "B0", "B1", "B2", "C1"]

STATE_DIM = 8
ACTION_DIM = 2

MODEL_DIR = "results/models"
LOG_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Initialize five independent DQN agents
# --------------------------------------------------

agents = {
    agent_id: DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_capacity=10000,
        target_update=100
    )
    for agent_id in AGENTS
}

env = TrafficSignalEnvironment(
    decision_interval=15,
    warmup_steps=20
)


# --------------------------------------------------
# 3. Training loop
# --------------------------------------------------

log_path = os.path.join(LOG_DIR, "training_log.csv")

try:

    with open(log_path, "w", newline="") as csvfile:

        fieldnames = [
            "episode",
            "decisions",
            "arrived_vehicles",
            "average_reward",
            "epsilon"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for episode in range(1, NUM_EPISODES + 1):

            observations = env.reset()

            episode_rewards = {
                agent_id: 0.0
                for agent_id in AGENTS
            }

            episode_losses = {
                agent_id: []
                for agent_id in AGENTS
            }

            total_arrived = 0
            decisions = 0

            print(f"\n{'=' * 50}")
            print(f"EPISODE {episode}/{NUM_EPISODES}")
            print(f"{'=' * 50}")

            # --------------------------------------
            # Episode interaction loop
            # --------------------------------------

            for decision in range(MAX_DECISIONS):

                actions = {}
                states = {}

                # Select one action per agent
                for agent_id in AGENTS:

                    state = np.array(
                        observations[agent_id]["normalized_state"],
                        dtype=np.float32
                    )

                    states[agent_id] = state

                    actions[agent_id] = (
                        agents[agent_id].select_action(state)
                    )

                # Execute joint actions in SUMO
                next_observations, rewards, info = env.step(actions)

                decisions += 1
                total_arrived += info["arrived_vehicles"]

                # Check whether the traffic simulation has ended
                done = (
                    traci.simulation.getMinExpectedNumber() == 0
                    or decision == MAX_DECISIONS - 1
                )

                # ----------------------------------
                # Store experiences and train agents
                # ----------------------------------

                for agent_id in AGENTS:

                    next_state = np.array(
                        next_observations[agent_id]["normalized_state"],
                        dtype=np.float32
                    )

                    reward = rewards[agent_id]
                    action = actions[agent_id]

                    agents[agent_id].remember(
                        states[agent_id],
                        action,
                        reward,
                        next_state,
                        done
                    )

                    loss = agents[agent_id].train_step()

                    if loss is not None:
                        episode_losses[agent_id].append(loss)

                    episode_rewards[agent_id] += reward

                observations = next_observations

                if done:
                    break

            # --------------------------------------
            # End of episode
            # --------------------------------------

            for agent_id in AGENTS:
                agents[agent_id].decay_epsilon()

            average_reward = np.mean(
                list(episode_rewards.values())
            )

            print(f"Decisions: {decisions}")
            print(f"Arrived vehicles: {total_arrived}")
            print(f"Average reward: {average_reward:.3f}")

            for agent_id in AGENTS:

                losses = episode_losses[agent_id]

                mean_loss = (
                    np.mean(losses)
                    if losses else 0.0
                )

                print(
                    f"{agent_id}: "
                    f"Reward={episode_rewards[agent_id]:.3f}, "
                    f"Epsilon={agents[agent_id].epsilon:.3f}, "
                    f"Mean loss={mean_loss:.4f}"
                )

            writer.writerow({
                "episode": episode,
                "decisions": decisions,
                "arrived_vehicles": total_arrived,
                "average_reward": average_reward,
                "epsilon": agents[AGENTS[0]].epsilon
            })

            csvfile.flush()

finally:
    env.close()


# --------------------------------------------------
# 4. Save trained models
# --------------------------------------------------

for agent_id in AGENTS:

    model_path = os.path.join(
        MODEL_DIR,
        f"{agent_id}_dqn.pth"
    )

    torch.save(
        agents[agent_id].policy_net.state_dict(),
        model_path
    )

    print(f"Saved {agent_id}: {model_path}")


print("\nTraining completed.")
