import sumolib

net = sumolib.net.readNet("SUMO/networks/2x2grid/2x2.net.xml")

for tls_id in ["1", "2", "5", "6"]:
    lanes = set()

    for edge in net.getEdges():
        for lane in edge.getLanes():
            for conn in lane.getOutgoing():
                if conn.getTLSID() == tls_id:
                    lanes.add(conn.getFromLane().getID())

    print(f"\nTraffic light {tls_id}")
    print(f"Number of incoming lanes: {len(lanes)}")

    for lane in sorted(lanes):
        print(" ", lane)

    print(f"Observation dimension: {4 + 1 + 2 * len(lanes)}")
