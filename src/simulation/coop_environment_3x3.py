
from src.simulation.marl_environment_3x3 import MARLEnvironment3x3


class CoopMARLEnvironment3x3(MARLEnvironment3x3):

    NEIGHBORS = {
        "A0": ["A1", "B0"],
        "A1": ["A0", "A2", "B1"],
        "A2": ["A1", "B2"],

        "B0": ["A0", "B1", "C0"],
        "B1": ["A1", "B0", "B2", "C1"],
        "B2": ["A2", "B1", "C2"],

        "C0": ["B0", "C1"],
        "C1": ["B1", "C0", "C2"],
        "C2": ["B2", "C1"],
    }

    ALPHA = 0.35

    def step(self, actions):

        observations, rewards, dones, infos = super().step(actions)

        local_rewards = {
            agent: float(rewards[agent])
            for agent in rewards
        }

        cooperative_rewards = {
            agent: (
                local_rewards[agent]
                + self.ALPHA * sum(
                    local_rewards[neighbor]
                    for neighbor in self.NEIGHBORS[agent]
                    if neighbor in local_rewards
                )
            )
            for agent in local_rewards
        }

        return (
            observations,
            cooperative_rewards,
            dones,
            infos
        )
