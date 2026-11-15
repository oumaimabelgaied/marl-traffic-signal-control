
import numpy as np

from sumo_rl.environment.env import SumoEnvironment


class MARLEnvironment2x2(SumoEnvironment):
    """
    Four-agent Cooperative DQN environment on a 2x2 SUMO grid.

    Agents: 1, 2, 5, 6
    Observation: 21 dimensions
    Action: 4 possible green phases
    Reward: local queue reward + weighted neighbors' rewards
    """

    AGENT_IDS = ["1", "2", "5", "6"]

    def __init__(
        self,
        delta_time=15,
        yellow_time=3,
        min_green=10,
        max_green=50,
        num_seconds=2000,
        sumo_seed=42,
        use_gui=False,
        fixed_ts=False,
        additional_sumo_cmd=None
    ):

        super().__init__(
            net_file="SUMO/networks/2x2grid/2x2.net.xml",
            route_file="SUMO/routes/2x2grid/2x2.rou.xml",
            use_gui=use_gui,
            begin_time=0,
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

        if set(self.ts_ids) != set(self.AGENT_IDS):
            raise ValueError(
                f"Unexpected traffic lights: {self.ts_ids}. "
                f"Expected: {self.AGENT_IDS}"
            )

        self.agents = self.AGENT_IDS.copy()

        self.unique_incoming_lanes = sorted(
            set(
                lane
                for agent in self.AGENT_IDS
                for lane in self.traffic_signals[agent].lanes
            )
        )

        for agent in self.agents:

            ts = self.traffic_signals[agent]

            if ts.num_green_phases != 4:
                raise ValueError(
                    f"Traffic light {agent}: expected 4 green phases, "
                    f"got {ts.num_green_phases}"
                )

            if ts.observation_space.shape != (21,):
                raise ValueError(
                    f"Traffic light {agent}: expected observation "
                    f"shape (21,), got {ts.observation_space.shape}"
                )

    def _sumo_step(self):
        """Advance SUMO by one second and collect queue and CO2 metrics."""

        super()._sumo_step()

        # Initialize histories if needed.
        if not hasattr(self, "queue_history"):
            self.queue_history = []

        if not hasattr(self, "co2_history"):
            self.co2_history = []

        # Queue collection.
        total_queue = sum(
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.unique_incoming_lanes
        )

        self.queue_history.append(total_queue)

        # CO2 collection.
        # SUMO returns emissions in mg/s for each active vehicle.
        active_vehicles = self.sumo.vehicle.getIDList()

        total_co2_mg_per_second = sum(
            self.sumo.vehicle.getCO2Emission(vehicle_id)
            for vehicle_id in active_vehicles
        )

        # One SUMO simulation step = one second.
        self.co2_history.append(total_co2_mg_per_second)

    @property
    def total_co2_mg(self):
        """Total simulated CO2 emissions in milligrams."""
        return float(sum(self.co2_history))

    @property
    def total_co2_kg(self):
        """Total simulated CO2 emissions in kilograms."""
        return self.total_co2_mg / 1_000_000

    def reset(self, seed=None, **kwargs):

        self.queue_history = []
        self.co2_history = []

        observations = super().reset(seed=seed, **kwargs)

        if set(observations.keys()) != set(self.agents):
            raise RuntimeError(
                f"Unexpected observations: {list(observations.keys())}"
            )

        return {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in self.agents
        }

    def step(self, actions):

        if set(actions.keys()) != set(self.agents):
            raise ValueError(
                f"Actions must be provided for agents {self.agents}"
            )

        for agent, action in actions.items():

            if not isinstance(action, (int, np.integer)):
                raise TypeError(
                    f"Action for {agent} must be an integer"
                )

            if not 0 <= action < 4:
                raise ValueError(
                    f"Invalid action {action} for traffic light {agent}"
                )

        observations, rewards, dones, infos = super().step(actions)

        observations = {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in self.agents
        }

        # Original local rewards.
        local_rewards = {
            agent: float(rewards[agent])
            for agent in self.agents
        }

        # Direct-neighbor topology.
        neighbors = {
            "1": ["2", "5"],
            "2": ["1", "6"],
            "5": ["1", "6"],
            "6": ["2", "5"],
        }

        # Original cooperation factor.
        alpha = 0.35

        # Cooperative reward: local reward + weighted neighbors' rewards.
        rewards = {
            agent: local_rewards[agent]
            + alpha * sum(
                local_rewards[neighbor]
                for neighbor in neighbors[agent]
            )
            for agent in self.agents
        }

        return observations, rewards, dones, infos


if __name__ == "__main__":

    env = MARLEnvironment2x2(
        delta_time=15,
        yellow_time=3,
        min_green=10,
        max_green=50,
        num_seconds=2000,
        sumo_seed=42,
        use_gui=False,
    )

    try:

        observations = env.reset(seed=42)

        print("\n=== INITIAL OBSERVATIONS ===")

        for agent, obs in observations.items():
            print(f"Agent {agent}: shape={obs.shape}")

        print("\n=== COOPERATIVE ACTION TEST ===")

        for step in range(5):

            actions = {
                agent: step % 4
                for agent in env.agents
            }

            observations, rewards, dones, infos = env.step(actions)

            print(f"\nDecision {step + 1}")
            print("Actions:", actions)
            print("Cooperative rewards:", rewards)
            print(f"Simulation time: {env.sim_step}")
            print(f"CO2: {env.total_co2_kg:.6f} kg")

            if dones.get("__all__", False):
                break

        print("\n=== FINAL STATISTICS ===")
        print(f"Queue samples: {len(env.queue_history)}")
        print(f"Mean queue: {np.mean(env.queue_history):.3f}")
        print(f"Total CO2: {env.total_co2_kg:.6f} kg")

    finally:
        env.close()
        print("\nEnvironment closed.")
