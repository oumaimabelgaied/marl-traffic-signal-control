import traci
import sumolib

NET = "SUMO/networks/2x2grid/2x2.net.xml"

net = sumolib.net.readNet(NET)

print("\n=== TRAFFIC LIGHTS ===")

for tls_id in ["1", "2", "5", "6"]:
    print(f"\n{'='*50}")
    print(f"TRAFFIC LIGHT: {tls_id}")

    connections = []

    for edge in net.getEdges():
        for lane in edge.getLanes():
            for conn in lane.getOutgoing():
                if conn.getTLSID() == tls_id:
                    connections.append(conn)

    connections.sort(key=lambda c: c.getTLLinkIndex())

    for conn in connections:
        print(
            f"Link {conn.getTLLinkIndex()}: "
            f"{conn.getFromLane().getID()} -> "
            f"{conn.getToLane().getID()} | "
            f"Direction: {conn.getDirection()}"
        )
