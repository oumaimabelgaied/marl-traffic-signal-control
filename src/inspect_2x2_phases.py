import xml.etree.ElementTree as ET

path = "SUMO/networks/2x2grid/2x2.net.xml"
root = ET.parse(path).getroot()

for tls in root.findall("tlLogic"):
    print(f"\nTraffic light {tls.get('id')}")

    for i, phase in enumerate(tls.findall("phase")):
        print(
            f"Phase {i}: "
            f"duration={phase.get('duration'):>3} s | "
            f"state={phase.get('state')}"
        )
