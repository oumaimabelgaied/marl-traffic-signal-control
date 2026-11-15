import traci


def apply_action(tls_id, action, green_duration=15):
    """
    Action 0: request green phase 0.
    Action 1: request green phase 2.

    Preserve yellow transitions and hold the selected green
    for the requested decision interval.
    """

    if action not in (0, 1):
        raise ValueError("Action must be 0 or 1.")

    current_phase = traci.trafficlight.getPhase(tls_id)
    target_phase = 0 if action == 0 else 2

    # Keep the requested green phase active
    if current_phase == target_phase:
        traci.trafficlight.setPhaseDuration(
            tls_id, green_duration
        )
        return f"{tls_id}: Holding green phase {target_phase}"

    # Switch from green 0 to green 2 through yellow 1
    if current_phase == 0 and target_phase == 2:
        traci.trafficlight.setPhase(tls_id, 1)
        return f"{tls_id}: Switching through yellow phase 1"

    # Switch from green 2 to green 0 through yellow 3
    if current_phase == 2 and target_phase == 0:
        traci.trafficlight.setPhase(tls_id, 3)
        return f"{tls_id}: Switching through yellow phase 3"

    # Never interrupt an ongoing yellow transition
    if current_phase in (1, 3):
        return f"{tls_id}: Yellow transition in progress; action deferred"

    raise RuntimeError(
        f"{tls_id}: Unexpected phase {current_phase}"
    )
