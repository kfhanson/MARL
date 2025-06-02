import os
import sys
import traci
import csv
import numpy as np
import random
import traceback

try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary, net
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")

SUMO_BINARY = checkBinary('sumo-gui')
CONFIG_FILE = "osm(1).sumocfg"
NET_FILE = "osm.net.xml"
TRAFFIC_LIGHT_ID = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'
OUTPUT_CSV_FILE = "sumo_data.csv"

# --- Parameters for the Data Collection Policy (iot_control.py like) ---
# These define how the *rule-based agent* makes decisions during data collection
CONGESTION_THRESHOLD_POLICY = 5 # vehicles
MIN_GREEN_TIME_IOT_POLICY = 10  # seconds
YELLOW_TIME_IOT_POLICY = 3      # seconds

# Mappings for the RULE-BASED POLICY's understanding of phases
# Action 0 -> N/S Green (SUMO Phase 0)
# Action 1 -> E/W Green (SUMO Phase 2)
# These MUST align with how your DQN agent will interpret actions 0 and 1.
AGENT_ACTION_TO_SUMO_GREEN_PHASE = {0: 0, 1: 2}
SUMO_GREEN_PHASE_TO_AGENT_ACTION = {v: k for k, v in AGENT_ACTION_TO_SUMO_GREEN_PHASE.items()}
SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3}

# --- State and Reward Definition for Logging (What the DQN will learn from) ---
# This defines which SUMO edges correspond to "North", "South", "East", "West"
# for the *state representation* logged in the CSV and used by the DQN.
# **ADJUST THESE TO YOUR ACTUAL SUMO EDGE IDs**
# North: Commuter
# East: Background / Other Commuter
# South: Commercial Hub (Approach B)
# West: Event Venue & Commercial (Approach A & C)
APPROACH_EDGES_FOR_STATE_LOGGING = {
    "north": "754598165#2",   # Commuter
    "south": "1053267667#3",  # Commercial Hub
    "east": "749662140#0",    # Background/Other Commuter
    "west": "885403818#2",    # Event Venue & Other Commercial
}

# Bins for discretizing halting counts for the logged state
VEHICLE_BINS_FOR_STATE_LOGGING = [5, 15, 30] # (0-4, 5-14, 15-29, 30+)

g_last_total_wait_time_log = 0
g_net_obj_policy_logger = None # Global for the network object

def discretize_value_log(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold: return i
    return len(bins)

def get_sumo_state_for_log():
    """ Returns (N,S,E,W) discretized halting counts for logging. """
    try:
        n = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["north"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        s = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["south"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        e = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["east"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        w = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["west"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        return (n, s, e, w)
    except Exception as e_state:
        print(f"Error in get_sumo_state_for_log: {e_state}")
        return None

def calculate_sumo_reward_for_log():
    global g_last_total_wait_time_log
    current_total = 0
    try:
        for edge_id in APPROACH_EDGES_FOR_STATE_LOGGING.values(): # Use the state logging edges
            current_total += traci.edge.getWaitingTime(edge_id)
        reward = (g_last_total_wait_time_log - current_total) / 100.0 # Normalize
        g_last_total_wait_time_log = current_total
        return reward
    except Exception as e_reward:
        print(f"Error in calculate_sumo_reward_for_log: {e_reward}")
        return 0

# --- Rule-Based Policy Logic ---
# This dictionary maps SUMO edge IDs to named directions AS UNDERSTOOD BY THE RULE-BASED POLICY.
# This might be the same as APPROACH_EDGES_FOR_STATE_LOGGING, or it could be different
# if your rule-based policy has a different internal naming/mapping.
# For consistency, let's assume it's the same for now, but be aware if your original iot_control.py used different names.
IOT_POLICY_EDGE_TO_DIRECTION_NAME = {
    "754598165#2": "north",
    "1053267667#3": "south",
    "749662140#0": "east",
    "885403818#2": "west",
}

def get_phase_green_directions_for_policy(tls_id, phase_state_str):
    """
    Determines which named directions get green for a given phase state string.
    Uses IOT_POLICY_EDGE_TO_DIRECTION_NAME for mapping.
    This should be replaced with your robust function from iot_control.py if it's more complex.
    """
    global g_net_obj_policy_logger
    if g_net_obj_policy_logger is None:
        print("Warning: Net object not loaded for phase direction mapping.")
        return set()

    green_directions = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        if not controlled_links: return green_directions

        for i, state_char in enumerate(phase_state_str):
            if state_char.lower() == 'g': # Consider 'G' and 'g' as green
                if i < len(controlled_links) and controlled_links[i]: # Check if link exists for this signal index
                    # A link is a list of tuples (fromLane, toLane, fromLinkIdx, toLinkIdx, internalLane)
                    # We need the fromLane of the first connection in this link
                    first_connection_in_link = controlled_links[i][0]
                    if first_connection_in_link:
                        from_lane_id = first_connection_in_link[0]
                        from_lane_obj = g_net_obj_policy_logger.getLane(from_lane_id)
                        if from_lane_obj:
                            from_edge_id = from_lane_obj.getEdge().getID()
                            if from_edge_id in IOT_POLICY_EDGE_TO_DIRECTION_NAME:
                                green_directions.add(IOT_POLICY_EDGE_TO_DIRECTION_NAME[from_edge_id])
    except KeyError as e_key: # Handles case where a lane ID might not be in the net object
        print(f"Warning: Lane ID {e_key} not found during phase direction mapping.")
    except Exception as e_phase_dir:
        print(f"Warning: Error in get_phase_green_directions_for_policy: {e_phase_dir}")
    return green_directions


def get_rule_based_policy_action(current_tl_sumo_phase_idx):
    """
    Implements the decision logic of your iot_control.py.
    Returns an agent action (0 or 1) or None if no specific change is dictated by rules.
    """
    try:
        # Get vehicle counts for rule-based policy's decision making
        # Using total vehicle number on the edge for policy decision
        policy_vehicle_counts = {}
        policy_congested_approaches = set()
        for edge_id, direction_name in IOT_POLICY_EDGE_TO_DIRECTION_NAME.items():
            count = traci.edge.getLastStepVehicleNumber(edge_id) # Or HaltingNumber if preferred by policy
            policy_vehicle_counts[direction_name] = count
            if count >= CONGESTION_THRESHOLD_POLICY:
                policy_congested_approaches.add(direction_name)

        program_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
        current_phase_definition = program_logic.phases[current_tl_sumo_phase_idx]
        current_phase_state_string = current_phase_definition.state

        # Determine which directions are green for the current SUMO phase
        directions_served_by_current_green = get_phase_green_directions_for_policy(TRAFFIC_LIGHT_ID, current_phase_state_string)

        is_current_green_serving_congestion = any(d in policy_congested_approaches for d in directions_served_by_current_green)

        if 'g' in current_phase_state_string.lower(): # If current SUMO phase is a green one
            current_agent_action = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx)

            if is_current_green_serving_congestion:
                # Rule: If current green serves congestion, policy might decide to stay/extend.
                # For discrete action logging, this means "take the current agent action".
                return current_agent_action
            elif policy_congested_approaches: # Current not serving congestion, but other approaches are
                # Rule: Policy decides to switch to serve other congestion.
                # Determine which *other* main agent action to take.
                if current_agent_action == 0: return 1 # If N/S green, switch to E/W (agent action 1)
                elif current_agent_action == 1: return 0 # If E/W green, switch to N/S (agent action 0)
                else: # Current green phase is not one of the main N/S or E/W, try to switch to one
                      # This part needs refinement: which one to pick if current is minor green?
                      # For now, let's default to action 0 if stuck in minor green with other congestion
                      return 0
            else:
                # No congestion anywhere, policy might decide to stay in current green.
                return current_agent_action
        else: # Current phase is yellow or all-red
            return None # Policy makes no decision during yellow/red, wait for next green opportunity
            
    except Exception as e_policy_action:
        print(f"Error in get_rule_based_policy_action: {e_policy_action}")
        return None # Default to no change on error


def run_sumo_and_log_data(num_simulation_steps=30000, data_collection_policy_name="rule_based"):
    global g_last_total_wait_time_log, g_net_obj_policy_logger

    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE] # CONFIG_FILE should use the new route.rou.xml
    sumo_cmd_list.extend([
        "--waiting-time-memory", "1000", "--time-to-teleport", "-1",
        "--no-step-log", "true", "--seed", str(random.randint(0, 100000))
    ])

    traci.start(sumo_cmd_list)
    g_net_obj_policy_logger = net.readNet(NET_FILE) # Load net for policy's phase mapping
    print(f"SUMO started for data collection with policy: {data_collection_policy_name}")

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
            phase_decision_timer = 0.0 # Time since current green phase started or decision was made
            
            # Initialize by running a few steps
            for _ in range(5): traci.simulationStep(); current_step+=1
            
            # s_t (state before action)
            state_s_for_log = get_sumo_state_for_log() 
            # The action 'a' taken at state_s_for_log
            action_applied_by_policy_for_log = -1 

            while traci.simulation.getMinExpectedNumber() > 0 and current_step < num_simulation_steps:
                current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                is_currently_green = current_tl_sumo_phase_idx in AGENT_ACTION_TO_SUMO_GREEN_PHASE.values()
                
                # Decision point: current phase is not green OR min green time has passed
                if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME_IOT_POLICY:
                    if state_s_for_log is None: # Ensure we have a valid previous state
                        state_s_for_log = get_sumo_state_for_log()
                        if state_s_for_log is None: # Still none, skip this iteration
                            traci.simulationStep(); current_step +=1; continue

                    # Get decision from the chosen policy
                    if data_collection_policy_name == "rule_based":
                        decided_agent_action = get_rule_based_policy_action(current_tl_sumo_phase_idx)
                    elif data_collection_policy_name == "random":
                        decided_agent_action = random.choice([0,1])
                    else: # Default
                        decided_agent_action = random.choice([0,1])

                    if decided_agent_action is not None:
                        action_applied_by_policy_for_log = decided_agent_action # This is 'a' for (s,a,r,s')
                        target_sumo_green_phase = AGENT_ACTION_TO_SUMO_GREEN_PHASE[decided_agent_action]

                        if current_tl_sumo_phase_idx != target_sumo_green_phase:
                            # Transition through yellow if currently on a main green phase
                            if current_tl_sumo_phase_idx in SUMO_GREEN_TO_YELLOW_PHASE:
                                yellow_phase_idx = SUMO_GREEN_TO_YELLOW_PHASE[current_tl_sumo_phase_idx]
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                                for _ in range(int(YELLOW_TIME_IOT_POLICY / traci.simulation.getDeltaT())):
                                    if traci.simulation.getMinExpectedNumber() == 0: break
                                    # Log intermediate steps if needed, or just advance
                                    traci.simulationStep(); current_step += 1 
                                if traci.simulation.getMinExpectedNumber() == 0: break
                            
                            # Set the target green phase
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                        phase_decision_timer = 0.0 # Reset timer for the new green phase
                    # If decided_agent_action is None (e.g. policy decides to do nothing during yellow)
                    # or if it's the same as current, action_applied_by_policy_for_log might stay -1 or be current
                    # We only want to log when an explicit agent action (0 or 1) is decided.
                    elif is_currently_green : # Policy decided to stay
                         action_applied_by_policy_for_log = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx, -1)


                # Step SUMO simulation (advances time by deltaT)
                traci.simulationStep()
                current_step += 1
                phase_decision_timer += traci.simulation.getDeltaT()
                
                # Observe results s', r, done AFTER the step
                state_s_prime_for_log = get_sumo_state_for_log()
                reward_for_log = calculate_sumo_reward_for_log() # Reward for the transition (s_t, a_t) -> s_{t+1}
                done_for_log = traci.simulation.getMinExpectedNumber() == 0 or current_step >= num_simulation_steps

                # Log if a valid (s, a, r, s') transition occurred due to a policy decision
                if state_s_for_log is not None and action_applied_by_policy_for_log != -1 and state_s_prime_for_log is not None:
                    row_to_write = list(state_s_for_log) + \
                                   [action_applied_by_policy_for_log, reward_for_log] + \
                                   list(state_s_prime_for_log) + \
                                   [done_for_log]
                    writer.writerow(row_to_write)
                    logged_transitions += 1
                
                state_s_for_log = state_s_prime_for_log # s' becomes s for the next iteration
                action_applied_by_policy_for_log = -1 # Reset, to be set only when a decision is made

                if done_for_log: break
            
            print(f"Data collection with '{data_collection_policy_name}' policy finished. Logged {logged_transitions} transitions to {OUTPUT_CSV_FILE}.")

    except traci.exceptions.FatalTraCIError as e: print(f"Fatal TraCI Error: {e}")
    except KeyboardInterrupt: print("Data collection interrupted.")
    except Exception as e:
        print(f"Unexpected Python error during data collection:")
        traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isEmbedded(): traci.close()

if __name__ == "__main__":
    run_sumo_and_log_data(num_simulation_steps=30000, data_collection_policy_name="rule_based")
    # run_sumo_and_log_data(num_simulation_steps=10000, data_collection_policy_name="random")