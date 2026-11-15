import traci
import sumolib

# Locate the SUMO executable
sumo_binary = sumolib.checkBinary("sumo")

# Simulation configuration
sumo_cmd = [
    sumo_binary,
    "-n", "SUMO/networks/test.net.xml",
    "-r", "SUMO/routes/test.rou.xml",
    "--no-step-log", "true",
    "--duration-log.disable", "true"
]

try:
    # Start SUMO and establish the TraCI connection
    traci.start(sumo_cmd)
    print("\n=== TraCI CONNECTION SUCCESSFUL ===")

    # Retrieve traffic light IDs
    tls_ids = traci.trafficlight.getIDList()
    print("\nTraffic lights:", tls_ids)

    # Inspect each traffic light
    for tls_id in tls_ids:
        print(f"\nTraffic light: {tls_id}")
        print("Current phase:", traci.trafficlight.getPhase(tls_id))
        print("Current state:", traci.trafficlight.getRedYellowGreenState(tls_id))

    # Advance the simulation
    print("\n=== SIMULATION ===")

    for step in range(50):
        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()

        if step % 10 == 0:
            print(
                f"Step {step + 1} | "
                f"Time: {traci.simulation.getTime():.1f}s | "
                f"Vehicles: {len(vehicles)}"
            )

    print("\n=== FINAL RESULTS ===")
    print("Simulation time:", traci.simulation.getTime())
    print("Active vehicles:", len(traci.vehicle.getIDList()))

finally:
    traci.close()
    print("\nTraCI connection closed.")
