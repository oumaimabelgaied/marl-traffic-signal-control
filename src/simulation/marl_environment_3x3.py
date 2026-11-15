
import os
import numpy as np

from sumo_rl.environment.env import SumoEnvironment


class MARLEnvironment3x3(SumoEnvironment):

    AGENT_IDS = [
        "A0", "A1", "A2",
        "B0", "B1", "B2",
        "C0", "C1", "C2"
    ]

    def __init__(
        self,
        delta_time=15,
        yellow_time=3,
        min_green=10,
        max_green=50,
        num_seconds=3600,
        sumo_seed=42,
        use_gui=False,
        fixed_ts=False,
        additional_sumo_cmd=None,
    ):

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        net_file = os.path.join(
            project_root, "SUMO", "networks", "3x3grid", "3x3.net.xml"
        )

        route_file = os.path.join(
            project_root, "SUMO", "routes", "3x3grid", "3x3.rou.xml"
        )

        super().__init__(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=None,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            single_agent=False,
            reward_fn="queue",
            sumo_seed=sumo_seed,
            fixed_ts=fixed_ts,
            add_system_info=False,
            add_per_agent_info=False,
            additional_sumo_cmd=additional_sumo_cmd,
        )

        self.ts_ids = self.AGENT_IDS.copy()
        self.agents = self.AGENT_IDS.copy()

        self.observations = {
            agent: None for agent in self.AGENT_IDS
        }
        self.rewards = {
            agent: None for agent in self.AGENT_IDS
        }

        self.unique_incoming_lanes = []
        self.queue_history = []

    def reset(self, seed=None, **kwargs):

        self.queue_history = []

        observations = super().reset(seed=seed, **kwargs)

        self.unique_incoming_lanes = sorted({
            lane
            for agent in self.AGENT_IDS
            for lane in self.sumo.trafficlight.getControlledLanes(agent)
        })

        return {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in self.AGENT_IDS
        }

    def _sumo_step(self):

        super()._sumo_step()

        total_queue = sum(
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.unique_incoming_lanes
        )

        self.queue_history.append(total_queue)

    def step(self, actions):

        if set(actions.keys()) != set(self.AGENT_IDS):
            raise ValueError(
                f"Expected actions for {self.AGENT_IDS}, "
                f"received {list(actions.keys())}"
            )

        for agent, action in actions.items():
            if not isinstance(action, (int, np.integer)):
                raise ValueError(f"Invalid action for {agent}: {action}")

            if action not in (0, 1):
                raise ValueError(f"Action must be 0 or 1 for {agent}")

        observations, rewards, dones, infos = super().step(actions)

        observations = {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in self.AGENT_IDS
            if agent in observations
        }

        rewards = {
            agent: float(rewards[agent])
            for agent in self.AGENT_IDS
            if agent in rewards
        }

        return observations, rewards, dones, infos
