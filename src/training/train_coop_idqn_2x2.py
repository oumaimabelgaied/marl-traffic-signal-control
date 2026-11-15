
import os
import sys
import csv
import random

import numpy as np
import torch

# Add src directory to Python import path
SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, SRC_DIR)

# Project imports
from agents.dqn_agent import DQNAgent
from simulation.coop_environment_2x2 import MARLEnvironment2x2


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42

NUM_EPISODES = 100
NUM_SECONDS = 3600

DELTA_TIME = 15
ALPHA = 0.35

AGENT_IDS = ["1", "2", "5", "6"]

RESULTS_DIR = "results/coop_idqn_2x2_training_100ep"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_FILE = os.path.join(RESULTS_DIR, "training_metrics.csv")


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# INITIALIZATION
# ==========================================

os.makedirs(MODELS_DIR, exist_ok=True)

env = MARLEnvironment2x2(
    delta_time=DELTA_TIME,
    yellow_time=3,
    min_green=10,
    max_green=50,
    num_seconds=NUM_SECONDS,
    sumo_seed=SEED,
    use_gui=False,
)

agents = {
    agent_id: DQNAgent(
        state_dim=21,
        action_dim=4,
    )
    for agent_id in AGENT_IDS
}


# ==========================================
# METRICS
# ==========================================

metrics_rows = []


# ==========================================
# COOPERATIVE DQN TRAINING
# ==========================================

try:

    for episode in range(NUM_EPISODES):

        observations = env.reset(seed=SEED + episode)

        episode_rewards = {
            agent_id: 0.0
            for agent_id in AGENT_IDS
        }

        episode_losses = {
            agent_id: []
            for agent_id in AGENT_IDS
        }

        done = False
        decision_count = 0

        print(f"\n{'=' * 45}")
        print(f"COOP-IDQN EPISODE {episode + 1}/{NUM_EPISODES}")
        print(f"{'=' * 45}")

        while not done:

            # 1. Each agent independently selects an action.
            actions = {
                agent_id: agents[agent_id].select_action(
                    observations[agent_id],
                    training=True,
                )
                for agent_id in AGENT_IDS
            }

            # 2. Apply all four actions in SUMO.
            next_observations, rewards, dones, infos = env.step(actions)

            # 3. Check whether the episode has ended.
            done = dones["__all__"]

            # 4. Store and train each agent using cooperative rewards.
            for agent_id in AGENT_IDS:

                agent = agents[agent_id]

                state = observations[agent_id]
                action = actions[agent_id]
                reward = rewards[agent_id]
                next_state = next_observations[agent_id]

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = agent.train_step()

                if loss is not None:
                    episode_losses[agent_id].append(loss)

                episode_rewards[agent_id] += reward

            observations = next_observations
            decision_count += 1

            if decision_count % 20 == 0:
                print(
                    f"Decision {decision_count} | "
                    f"Simulation time: {infos['step']:.0f}s"
                )

        # 5. Decay exploration after each episode.
        for agent in agents.values():
            agent.decay_epsilon()

        # 6. Calculate actual network queue metrics.
        average_network_queue = (
            float(np.mean(env.queue_history))
            if env.queue_history
            else 0.0
        )

        maximum_network_queue = (
            float(np.max(env.queue_history))
            if env.queue_history
            else 0.0
        )

        print("\nEpisode results:")
        print(
            f"Network average queue: {average_network_queue:.3f}"
        )
        print(
            f"Network maximum queue: {maximum_network_queue:.0f}"
        )

        # 7. Record per-agent metrics.
        for agent_id in AGENT_IDS:

            agent = agents[agent_id]

            mean_cooperative_reward = (
                episode_rewards[agent_id] / decision_count
                if decision_count > 0
                else 0.0
            )

            losses = episode_losses[agent_id]

            mean_loss = (
                float(np.mean(losses))
                if losses
                else None
            )

            row = {
                "episode": episode + 1,
                "agent": agent_id,
                "episode_return": episode_rewards[agent_id],
                "mean_cooperative_reward_at_decision":
                    mean_cooperative_reward,
                "mean_loss": mean_loss,
                "epsilon": agent.epsilon,
                "decisions": decision_count,
                "average_network_queue": average_network_queue,
                "maximum_network_queue": maximum_network_queue,
            }

            metrics_rows.append(row)

            print(
                f"Agent {agent_id} | "
                f"Return: {episode_rewards[agent_id]:.2f} | "
                f"Mean cooperative reward: "
                f"{mean_cooperative_reward:.2f} | "
                f"Loss: {mean_loss} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Save metrics after every episode.
        with open(
            METRICS_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            fieldnames = [
                "episode",
                "agent",
                "episode_return",
                "mean_cooperative_reward_at_decision",
                "mean_loss",
                "epsilon",
                "decisions",
                "average_network_queue",
                "maximum_network_queue",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(metrics_rows)

finally:
    env.close()


# ==========================================
# SAVE TRAINED AGENTS
# ==========================================

for agent_id, agent in agents.items():

    model_path = os.path.join(
        MODELS_DIR,
        f"coop_dqn_agent_{agent_id}.pth",
    )

    torch.save(
        {
            "policy_net": agent.policy_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "learning_steps": agent.learning_steps,
        },
        model_path,
    )

    print(f"Saved cooperative agent {agent_id}: {model_path}")


print("\nCoop-IDQN training completed.")
print(f"Metrics saved to: {METRICS_FILE}")
