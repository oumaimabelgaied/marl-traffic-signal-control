import traci
import sumolib

from traffic_light_controller import apply_action

SUMO_CMD = [
    sumolib.checkBinary("sumo"),
    "-n", "SUMO/networks/test3.net.xml",
    "-r", "SUMO/routes/test3.rou.xml",
    "--no-step-log", "true",
    "--duration-log.disable", "true"
]

RL_AGENTS = ["A1", "B0", "B1", "B2", "C1"]


def show_phases(label):
    print(f"\n{label}")
    for tls_id in RL_AGENTS:
        print(tls_id, "Phase:", traci.trafficlight.getPhase(tls_id))


try:
    traci.start(SUMO_CMD)

    for _ in range(20):
        traci.simulationStep()

    show_phases("INITIAL PHASES")

    # Action 1: request green phase 2
    print("\n=== ACTION 1: GREEN PHASE 2 ===")

    for tls_id in RL_AGENTS:
        print(apply_action(tls_id, 1))

    for _ in range(4):
        traci.simulationStep()

    show_phases("AFTER TRANSITION TO PHASE 2")

    # Action 0: request green phase 0
    print("\n=== ACTION 0: GREEN PHASE 0 ===")

    for tls_id in RL_AGENTS:
        print(apply_action(tls_id, 0))

    for _ in range(4):
        traci.simulationStep()

    show_phases("AFTER TRANSITION TO PHASE 0")

    # Request the already active phase
    print("\n=== REPEATED ACTION TEST ===")

    for tls_id in RL_AGENTS:
        print(apply_action(tls_id, 0))

finally:
    traci.close()
    print("\nSimulation closed.")
