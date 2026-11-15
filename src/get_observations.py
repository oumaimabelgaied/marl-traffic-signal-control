import traci
import sumolib

SUMO_CMD = [
    sumolib.checkBinary("sumo"),
    "-n", "SUMO/networks/test3.net.xml",
    "-r", "SUMO/routes/test3.rou.xml",
    "--no-step-log", "true",
    "--duration-log.disable", "true"
]

RL_AGENTS = ["A1", "B0", "B1", "B2", "C1"]


def get_intersection_observation(tls_id):
    # Get incoming lanes from the traffic light's controlled links
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
        vehicle_count += traci.lane.getLastStepVehicleNumber(lane_id)
        queue_length += traci.lane.getLastStepHaltingNumber(lane_id)

        lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)
        lane_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)

        total_speed += lane_speed * len(lane_vehicles)

        for vehicle_id in lane_vehicles:
            total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
    average_speed = (
        total_speed / vehicle_count if vehicle_count > 0 else 0.0
    )

    return {
        "intersection": tls_id,
        "phase": traci.trafficlight.getPhase(tls_id),
        "incoming_lanes": incoming_lanes,
        "vehicle_count": vehicle_count,
        "queue_length": queue_length,
        "average_speed": round(average_speed, 2),
        "total_waiting_time": round(total_waiting_time, 2)
    }


try:
    traci.start(SUMO_CMD)

    # Let traffic enter the network before collecting observations
    for _ in range(20):
        traci.simulationStep()

    print("\n========== INTERSECTION OBSERVATIONS ==========")

    for tls_id in RL_AGENTS:
        observation = get_intersection_observation(tls_id)

        print(f"\nIntersection: {observation['intersection']}")
        print(f"Incoming lanes: {observation['incoming_lanes']}")
        print(f"Current phase: {observation['phase']}")
        print(f"Vehicle count: {observation['vehicle_count']}")
        print(f"Queue length: {observation['queue_length']}")
        print(f"Average speed: {observation['average_speed']} m/s")
        print(f"Total waiting time: {observation['total_waiting_time']} s")

finally:
    traci.close()
