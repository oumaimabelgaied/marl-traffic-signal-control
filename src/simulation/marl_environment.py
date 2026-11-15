import traci
import sumolib
import math

from traffic_light_controller import apply_action


class TrafficSignalEnvironment:

    
    def __init__(
        self,
        decision_interval=15,
        warmup_steps=20,
        route_file="SUMO/routes/test3.rou.xml",
        seed=42
    ):
        self.agents = ["A1", "B0", "B1", "B2", "C1"]

        self.decision_interval = decision_interval
        self.warmup_steps = warmup_steps

        self.route_file = str(route_file)
        self.seed = seed

        self.sumo_cmd = [
            "sumo",
            "-n", "SUMO/networks/test3.net.xml",
            "-r", self.route_file,
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--seed", str(self.seed)
        ]

        self.started = False


    def normalize_state(self, state):
        return [
            min(state[0] / 20.0, 1.0),
            min(state[1] / 10.0, 1.0),
            min(state[2] / 13.89, 1.0),
            min(math.log1p(state[3]) / math.log1p(100), 1.0),
            *state[4:]
        ]

    def reset(self):

        if self.started:
            self.close()

        traci.start(self.sumo_cmd)
        self.started = True

        for _ in range(self.warmup_steps):
            traci.simulationStep()

        return self.get_observations()
    

    def get_queue_length(self, tls_id):
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        incoming_lanes = list(dict.fromkeys(
            link[0]
            for signal_group in controlled_links
            for link in signal_group
            if link
        ))

        return sum(
            traci.lane.getLastStepHaltingNumber(lane_id)
            for lane_id in incoming_lanes
        )

    def get_agent_observation(self, tls_id):

        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        incoming_lanes = list(dict.fromkeys(
            link[0]
            for signal_group in controlled_links
            for link in signal_group
            if link
        ))

        vehicle_count = 0
        queue_length = 0
        total_speed = 0.0
        total_waiting_time = 0.0

        for lane_id in incoming_lanes:

            lane_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)

            vehicle_count += len(lane_vehicles)

            queue_length += (
                traci.lane.getLastStepHaltingNumber(lane_id)
            )

            lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)

            total_speed += lane_speed * len(lane_vehicles)

            for vehicle_id in lane_vehicles:
                total_waiting_time += (
                    traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                )

        average_speed = (
            total_speed / vehicle_count
            if vehicle_count > 0 else 0.0
        )

        phase = traci.trafficlight.getPhase(tls_id)

        # One-hot encoding of the four signal phases
        phase_encoding = [0, 0, 0, 0]
        phase_encoding[phase] = 1

        state = [
            vehicle_count,
            queue_length,
            round(average_speed, 2),
            round(total_waiting_time, 2)
        ] + phase_encoding

        return {
            "state": state,
            "normalized_state": self.normalize_state(state),
            "vehicle_count": vehicle_count,
            "queue_length": queue_length,
            "average_speed": round(average_speed, 2),
            "total_waiting_time": round(total_waiting_time, 2),
            "phase": phase
        }

    def get_observations(self):

        return {
            agent: self.get_agent_observation(agent)
            for agent in self.agents
        }

   
    def step(self, actions):
        if set(actions.keys()) != set(self.agents):
            raise ValueError("Actions must be provided for all five RL agents.")

        for agent in self.agents:
            apply_action(
                agent,
                actions[agent],
                self.decision_interval + 3
            )

        total_arrived = 0
        elapsed_steps = 0

        # Accumulate queue length at every actual simulation second
        queue_area = {agent: 0.0 for agent in self.agents}

        for _ in range(self.decision_interval):

            traci.simulationStep()
            elapsed_steps += 1

            total_arrived += traci.simulation.getArrivedNumber()

            for agent in self.agents:
                queue_area[agent] += self.get_queue_length(agent)

            # Stop immediately when no vehicles remain
            if traci.simulation.getMinExpectedNumber() == 0:
                break

        # Calculate reward using the actual elapsed duration
        rewards = {
            agent: (
                -queue_area[agent] / elapsed_steps
                if elapsed_steps > 0 else 0.0
            )
            for agent in self.agents
        }

        observations = self.get_observations()

        info = {
            "simulation_time": traci.simulation.getTime(),
            "arrived_vehicles": total_arrived,
            "elapsed_steps": elapsed_steps,
            "queue_area": queue_area
        }

        return observations, rewards, info


    def close(self):

        if self.started:
            traci.close()
            self.started = False


if __name__ == "__main__":

    env = TrafficSignalEnvironment()

    try:

        observations = env.reset()

        print("\n=== INITIAL OBSERVATIONS ===")

        for agent, observation in observations.items():
            print(
                f"{agent}: state={observation['state']}, "
                f"phase={observation['phase']}"
            )
            print(
                f"{agent}: normalized_state="
                f"{observation['normalized_state']}"
            )

        print("\n=== RANDOM MULTI-AGENT TEST ===")

        for step in range(5):

            # Temporary random actions for testing
            actions = {
                agent: (step + index) % 2
                for index, agent in enumerate(env.agents)
            }

            print(f"\nDecision step {step + 1}")
            print("Actions:", actions)

            observations, rewards, info = env.step(actions)

            print("Simulation time:", info["simulation_time"])
            print("Arrived vehicles:", info["arrived_vehicles"])
            print("Rewards:", rewards)


            for agent, observation in observations.items():

                print(
                    f"{agent}: "
                    f"phase={observation['phase']}, "
                    f"vehicles={observation['vehicle_count']}, "
                    f"queue={observation['queue_length']}, "
                    f"state={observation['state']}"
                )

    finally:
        env.close()
        print("\nEnvironment closed.")
