import os
import sys
import traci
import csv
import numpy as np
import random
import traceback

# --- SUMO Configuration ---
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary, net # Import net for iot_control logic
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")

SUMO_BINARY = checkBinary('sumo')
CONFIG_FILE = "osm.sumocfg"
NET_FILE = "osm.net.xml" # Needed for iot_control phase mapping
TRAFFIC_LIGHT_ID = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'
OUTPUT_CSV_FILE = "sumo_offline_data_rule_based.csv"

# --- Parameters from iot_control.py (for data collection policy) ---
COMMUNICATION_RANGE_POLICY = 100.0
CONGESTION_THRESHOLD_POLICY = 5
# Note: iot_control.py adjusts durations. For logging (s,a,r,s'), we need to map this to
# a discrete action (e.g., stay in current green phase or switch).
# We'll simplify this for now: if iot_control would extend, we consider it "staying".
# If it would switch due to congestion elsewhere, or for an EV, that's a "switch" action.

MIN_GREEN_TIME_IOT_POLICY = 10 # Min time before iot_control logic might adjust/switch
YELLOW_TIME_IOT_POLICY = 3
# These mappings define which agent action (0 or 1) corresponds to which main green phase
# This MUST align with how your offline DQN agent will interpret actions.
# Action 0 -> N/S Green (SUMO Phase 0)
# Action 1 -> E/W Green (SUMO Phase 2)
AGENT_ACTION_TO_SUMO_GREEN_PHASE = {0: 0, 1: 2}
SUMO_GREEN_PHASE_TO_AGENT_ACTION = {v: k for k, v in AGENT_ACTION_TO_SUMO_GREEN_PHASE.items()}
SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3}


# --- State and Reward Definition (consistent with DQN) ---
APPROACH_EDGES_FOR_STATE = { # Edges used for the S,A,R,S' logging
    "north": "754598165#2",
    "south": "1053267667#3",
    "east": "749662140#0",
    "west": "885403818#2",
}
VEHICLE_BINS_FOR_STATE = [5, 15, 30]

g_last_total_wait_time_log = 0

def discretize_value_log(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold: return i
    return len(bins)

def get_sumo_state_for_log(): # Returns (N,S,E,W) discretized counts
    try:
        n = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["north"]), VEHICLE_BINS_FOR_STATE)
        s = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["south"]), VEHICLE_BINS_FOR_STATE)
        e = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["east"]), VEHICLE_BINS_FOR_STATE)
        w = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["west"]), VEHICLE_BINS_FOR_STATE)
        return (n, s, e, w)
    except Exception: return None

def calculate_sumo_reward_for_log():
    global g_last_total_wait_time_log
    current_total = 0
    try:
        for edge_id in APPROACH_EDGES_FOR_STATE.values():
            current_total += traci.edge.getWaitingTime(edge_id)
        reward = (g_last_total_wait_time_log - current_total) / 100.0
        g_last_total_wait_time_log = current_total
        return reward
    except Exception: return 0

# --- Simplified Logic from iot_control.py for decision making ---
# This needs the `net_obj` and `get_phase_state_directions` from your `iot_control.py`
# For simplicity, I'll create a placeholder version.
# You should integrate your actual functions if they are more complex.
g_net_obj_policy = None # Will be loaded once

# Placeholder for actual approach edges used by iot_control's phase direction mapping
# This should match the keys used in your original `get_phase_state_directions`
IOT_CONTROL_APPROACH_EDGES_FOR_PHASE_MAPPING = {
    "754598165#2": "north", # Example mapping, adjust to your iot_control.py
    "1053267667#3": "south",
    "749662140#0": "east",
    "885403818#2": "west",
}

def get_phase_state_directions_policy(tls_id, phase_state_str):
    """Simplified version. Integrate your actual function from iot_control.py."""
    global g_net_obj_policy
    if g_net_obj_policy is None: return set() # Should not happen after loading

    green_directions = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        if not controlled_links: return green_directions
        for i, state_char in enumerate(phase_state_str):
            if state_char.lower() == 'g':
                if i < len(controlled_links) and controlled_links[i]:
                    first_link_tuple = controlled_links[i][0]
                    if first_link_tuple:
                        from_lane_id = first_link_tuple[0]
                        from_lane = g_net_obj_policy.getLane(from_lane_id) # Use the global net object
                        if from_lane:
                            from_edge_id = from_lane.getEdge().getID()
                            if from_edge_id in IOT_CONTROL_APPROACH_EDGES_FOR_PHASE_MAPPING:
                                green_directions.add(IOT_CONTROL_APPROACH_EDGES_FOR_PHASE_MAPPING[from_edge_id])
    except Exception as e:
        print(f"Warning: Error in get_phase_state_directions_policy: {e}")
    return green_directions


def get_iot_control_decision_policy(current_sumo_phase_idx):
    """
    Makes a decision based on simplified iot_control.py logic.
    Returns an agent action (0 or 1) or None if no change.
    """
    # --- Get vehicle counts for iot_control logic (might differ from state logging) ---
    vehicle_counts_iot = {}
    congested_approaches_iot = set()
    try:
        for edge_id, direction_name in IOT_CONTROL_APPROACH_EDGES_FOR_PHASE_MAPPING.items():
            # Using getLastStepVehicleNumber for iot_control as it might use total count
            count = traci.edge.getLastStepVehicleNumber(edge_id)
            vehicle_counts_iot[direction_name] = count
            if count >= CONGESTION_THRESHOLD_POLICY:
                congested_approaches_iot.add(direction_name)

        # --- iot_control.py like logic ---
        program_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
        current_phase_def = program_logic.phases[current_sumo_phase_idx]
        current_phase_state_str = current_phase_def.state

        current_green_directions_iot = get_phase_state_directions_policy(TRAFFIC_LIGHT_ID, current_phase_state_str)
        is_current_phase_for_congested_iot = any(d in congested_approaches_iot for d in current_green_directions_iot)

        # Simplified: If current green serves congestion, stay (return current agent action).
        # If current green does NOT serve congestion BUT other approaches ARE congested, switch.
        if 'g' in current_phase_state_str.lower(): # If current SUMO phase is green
            if is_current_phase_for_congested_iot:
                 # Stay in current green phase (iot_control would extend)
                return SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_sumo_phase_idx)
            elif congested_approaches_iot: # Current not for congested, but others are
                # Switch to the *other* main green phase
                current_agent_action = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_sumo_phase_idx)
                if current_agent_action == 0: return 1 # If N/S, switch to E/W
                elif current_agent_action == 1: return 0 # If E/W, switch to N/S
                else: return None # Unmapped green phase, no clear switch
            else: # No congestion anywhere, stay
                return SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_sumo_phase_idx)
        return None # Not a green phase or no clear decision
    except Exception as e:
        print(f"Error in iot_control_decision_policy: {e}")
        return None # Default to no change on error

# --- Main Data Collection Function ---
def run_sumo_and_log_data_iot_policy(num_simulation_steps=10000):
    global g_last_total_wait_time_log, g_net_obj_policy
    
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE]
    sumo_cmd_list.extend([
        "--waiting-time-memory", "1000", "--time-to-teleport", "-1",
        "--no-step-log", "true", "--seed", str(random.randint(0, 100000))
    ])

    traci.start(sumo_cmd_list)
    g_net_obj_policy = net.readNet(NET_FILE) # Load net file for phase mapping
    print(f"SUMO started for data collection with iot_control.py-like policy.")

    csv_header = [
        "state_N", "state_S", "state_E", "state_W", "action", "reward",
        "next_state_N", "next_state_S", "next_state_E", "next_state_W", "done"
    ]
    logged_transitions = 0

    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            current_step = 0
            g_last_total_wait_time_log = 0
            phase_decision_timer = 0.0
            
            # Initialize: Run a few steps to get an initial state
            for _ in range(5): traci.simulationStep(); current_step+=1
            
            last_logged_state_tuple = get_sumo_state_for_log()
            # The "action_taken_to_reach_last_logged_state" is unknown for the very first state.
            # We will log a transition *after* an action is taken by the iot_policy.
            action_applied_by_iot_policy = -1


            while traci.simulation.getMinExpectedNumber() > 0 and current_step < num_simulation_steps:
                sim_time = traci.simulation.getTime()
                current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                is_currently_green = current_tl_sumo_phase_idx in AGENT_ACTION_TO_SUMO_GREEN_PHASE.values()
                
                # Store the state *before* the iot_policy makes a decision for this step
                state_s_for_log = get_sumo_state_for_log()
                
                # Decision point for iot_control_policy
                if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME_IOT_POLICY:
                    # Get decision from iot_control logic
                    # This returns the *desired agent action* (0 or 1)
                    decided_agent_action = get_iot_control_decision_policy(current_tl_sumo_phase_idx)

                    if decided_agent_action is not None:
                        action_applied_by_iot_policy = decided_agent_action # This is 'a' for (s,a,r,s')
                        target_sumo_green_phase = AGENT_ACTION_TO_SUMO_GREEN_PHASE[decided_agent_action]

                        if current_tl_sumo_phase_idx != target_sumo_green_phase:
                            if current_tl_sumo_phase_idx in SUMO_GREEN_TO_YELLOW_PHASE:
                                yellow_phase_idx = SUMO_GREEN_TO_YELLOW_PHASE[current_tl_sumo_phase_idx]
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                                for _ in range(int(YELLOW_TIME_IOT_POLICY / traci.simulation.getDeltaT())):
                                    if traci.simulation.getMinExpectedNumber() == 0: break
                                    traci.simulationStep(); current_step += 1
                                if traci.simulation.getMinExpectedNumber() == 0: break
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                            else:
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                            phase_decision_timer = 0.0
                    # If decided_agent_action is None, iot_control policy made no change, let current phase continue
                    # but we still need to log an "action" if a decision point was reached.
                    # For logging, if no explicit switch, assume "stay in current agent action"
                    elif is_currently_green: # If it was green and policy chose to stay
                        action_applied_by_iot_policy = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx, -1)


                # Step SUMO simulation
                traci.simulationStep()
                current_step += 1
                if is_currently_green: # Increment timer if it was green
                    phase_decision_timer += traci.simulation.getDeltaT()
                
                # Observe results s', r, done
                state_s_prime_for_log = get_sumo_state_for_log()
                reward_for_log = calculate_sumo_reward_for_log()
                done_for_log = traci.simulation.getMinExpectedNumber() == 0 or current_step >= num_simulation_steps

                # Log if a valid (s, a, r, s') transition occurred
                if state_s_for_log is not None and action_applied_by_iot_policy != -1 and state_s_prime_for_log is not None:
                    row_to_write = list(state_s_for_log) + \
                                   [action_applied_by_iot_policy, reward_for_log] + \
                                   list(state_s_prime_for_log) + \
                                   [done_for_log]
                    writer.writerow(row_to_write)
                    logged_transitions += 1
                    action_applied_by_iot_policy = -1 # Reset for next valid logging

                if done_for_log: break

            print(f"Data collection with iot_control policy finished. Logged {logged_transitions} transitions to {OUTPUT_CSV_FILE}.")

    except traci.exceptions.FatalTraCIError as e: print(f"Fatal TraCI Error: {e}")
    except KeyboardInterrupt: print("Data collection interrupted.")
    except Exception as e:
        print(f"Unexpected Python error:")
        traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isEmbedded(): traci.close()


if __name__ == "__main__":
    run_sumo_and_log_data_iot_policy(num_simulation_steps=20000) # Collect 20k steps for example