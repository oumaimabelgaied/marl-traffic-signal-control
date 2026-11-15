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

try:
    traci.start(SUMO_CMD)

    # Advance simulation
    for _ in range(20):
        traci.simulationStep()

    print("\n=== INITIAL PHASES ===")

    for tls_id in RL_AGENTS:
        print(tls_id, "Phase:", traci.trafficlight.getPhase(tls_id))

    # Request the opposite green phase through yellow
    print("\n=== REQUESTING PHASE TRANSITIONS ===")

    for tls_id in RL_AGENTS:
        current_phase = traci.trafficlight.getPhase(tls_id)

        if current_phase == 0:
            traci.trafficlight.setPhase(tls_id, 1)
            print(tls_id, ": Green 0 -> Yellow 1 -> Green 2")

        elif current_phase == 2:
            traci.trafficlight.setPhase(tls_id, 3)
            print(tls_id, ": Green 2 -> Yellow 3 -> Green 0")

        else:
            print(tls_id, ": Already in a transition phase")

    # Observe the transition over 6 seconds
    print("\n=== TRANSITION MONITORING ===")

    for step in range(6):
        traci.simulationStep()

        print(f"\nSimulation time: {traci.simulation.getTime()} s")

        for tls_id in RL_AGENTS:
            phase = traci.trafficlight.getPhase(tls_id)
            state = traci.trafficlight.getRedYellowGreenState(tls_id)

            print(f"{tls_id}: Phase {phase} | Signal: {state}")

finally:
    traci.close()
    print("\nSimulation closed.")
