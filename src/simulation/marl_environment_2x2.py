
import numpy as np

from sumo_rl.environment.env import SumoEnvironment


class MARLEnvironment2x2(SumoEnvironment):
    """
    Four-agent Independent DQN environment on a 2x2 SUMO grid.

    Agents:
        1, 2, 5, 6

    Each agent:
        Observation: 21 dimensions
        Action: 4 possible green phases
        Reward: negative instantaneous number of halted vehicles

    Additional metrics:
        - Queue history
        - Instantaneous CO2 emissions (mg/s)
        - Accumulated CO2 emissions (mg)
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

        # Verify that the network contains exactly our four agents.
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

        # Verify action and observation dimensions.
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
        """
        Advance SUMO by one second and collect:
        - Total queue
        - Total instantaneous CO2 emissions

        SUMO getCO2Emission() returns mg/s.
        With the default one-second simulation step,
        each measurement is accumulated over one second.
        """

        super()._sumo_step()

        # Initialize histories if necessary.
        if not hasattr(self, "queue_history"):
            self.queue_history = []

        if not hasattr(self, "co2_history"):
            self.co2_history = []

        # 1. Collect total queue on unique incoming lanes.
        total_queue = sum(
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.unique_incoming_lanes
        )

        self.queue_history.append(total_queue)

        # 2. Collect instantaneous CO2 emissions.
        active_vehicles = self.sumo.vehicle.getIDList()

        step_co2_mg = sum(
            self.sumo.vehicle.getCO2Emission(vehicle_id)
            for vehicle_id in active_vehicles
        )

        # SUMO's default simulation step is 1 second.
        # Therefore: mg/s * 1 second = mg.
        self.co2_history.append(step_co2_mg)

    def reset(self, seed=None, **kwargs):

        # Reset metrics for the new episode.
        self.queue_history = []
        self.co2_history = []

        observations = super().reset(seed=seed, **kwargs)

        # Ensure all four agents are represented.
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

        # Return consistent observations for the four agents.
        observations = {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in self.agents
        }

        # SUMO-RL's queue reward is negative total halted vehicles.
        rewards = {
            agent: float(rewards[agent])
            for agent in self.agents
        }

        return observations, rewards, dones, infos

    @property
    def total_co2_mg(self):
        """Total accumulated CO2 emissions in milligrams."""
        return float(sum(self.co2_history))

    @property
    def total_co2_kg(self):
        """Total accumulated CO2 emissions in kilograms."""
        return self.total_co2_mg / 1_000_000.0


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
            print(
                f"Agent {agent}: "
                f"shape={obs.shape}, "
                f"observation={obs}"
            )

        print("\n=== RANDOM ACTION TEST ===")

        for step in range(5):

            actions = {
                agent: step % 4
                for agent in env.agents
            }

            observations, rewards, dones, infos = env.step(actions)

            print(f"\nDecision {step + 1}")
            print("Actions:", actions)
            print("Rewards:", rewards)

            print(
                f"Simulation time: {env.sim_step}s | "
                f"Queue: {env.queue_history[-1]} | "
                f"CO2: {env.total_co2_kg:.6f} kg"
            )

            if dones.get("__all__", False):
                print("Episode finished.")
                break

        print("\n=== FINAL METRICS ===")
        print("Queue samples:", len(env.queue_history))
        print("Mean queue:", np.mean(env.queue_history))
        print("Maximum queue:", np.max(env.queue_history))
        print("CO2 total (mg):", env.total_co2_mg)
        print("CO2 total (kg):", env.total_co2_kg)

    finally:
        env.close()
        print("\nEnvironment closed.")
