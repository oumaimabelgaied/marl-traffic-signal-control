import traci
import sumolib

cmd = [
    sumolib.checkBinary("sumo"),
    "-n", "SUMO/networks/test3.net.xml",
    "-r", "SUMO/routes/test3.rou.xml",
    "--no-step-log", "true"
]

try:
    traci.start(cmd)

    for tls_id in traci.trafficlight.getIDList():
        print(f"\nIntersection: {tls_id}")

        for program in traci.trafficlight.getAllProgramLogics(tls_id):
            print("Program:", program.programID)
            print("Number of phases:", len(program.phases))

            for i, phase in enumerate(program.phases):
                print(
                    f"  Phase {i}: "
                    f"duration={phase.duration}, "
                    f"state={phase.state}"
                )
finally:
    traci.close()
