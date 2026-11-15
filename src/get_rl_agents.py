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

    rl_agents = []
    fixed_intersections = []

    for tls_id in traci.trafficlight.getIDList():
        programs = traci.trafficlight.getAllProgramLogics(tls_id)

        if not programs:
            continue

        active_program = next(
            (p for p in programs
             if p.programID == traci.trafficlight.getProgram(tls_id)),
            programs[0]
        )

        phase_count = len(active_program.phases)

        if phase_count > 1:
            rl_agents.append(tls_id)
        else:
            fixed_intersections.append(tls_id)

    print("\n=== RL AGENTS ===")
    print("Number of agents:", len(rl_agents))
    print("Agent IDs:", rl_agents)

    print("\n=== SINGLE-PHASE INTERSECTIONS ===")
    print("Intersection IDs:", fixed_intersections)

finally:
    traci.close()
