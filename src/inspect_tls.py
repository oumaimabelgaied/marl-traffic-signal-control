import traci
import sumolib

sumo_cmd = [
    sumolib.checkBinary("sumo"),
    "-n", "SUMO/networks/test.net.xml",
    "-r", "SUMO/routes/test.rou.xml",
    "--no-step-log", "true",
    "--duration-log.disable", "true"
]

try:
    traci.start(sumo_cmd)

    for tls_id in traci.trafficlight.getIDList():
        print(f"\n{'=' * 40}")
        print(f"TRAFFIC LIGHT: {tls_id}")

        programs = traci.trafficlight.getAllProgramLogics(tls_id)

        for program in programs:
            print(f"\nProgram ID: {program.programID}")
            print(f"Number of phases: {len(program.phases)}")

            for i, phase in enumerate(program.phases):
                print(
                    f"Phase {i}: "
                    f"duration={phase.duration}s | "
                    f"state={phase.state}"
                )

    print("\n=== OBSERVING PHASE CHANGES ===")

    for step in range(100):
        traci.simulationStep()

        if step % 5 == 0:
            print(f"\nTime: {traci.simulation.getTime()}s")

            for tls_id in traci.trafficlight.getIDList():
                print(
                    f"{tls_id}: "
                    f"phase={traci.trafficlight.getPhase(tls_id)}, "
                    f"state={traci.trafficlight.getRedYellowGreenState(tls_id)}"
                )

finally:
    traci.close()
