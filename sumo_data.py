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
        from sumolib import checkBinary, net
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")

SUMO_BINARY = checkBinary('sumo-gui')
CONFIG_FILE = "osm.sumocfg"
NET_FILE = "osm.net.xml"
TRAFFIC_LIGHT_ID = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339' # Your TL ID from the image
OUTPUT_CSV_FILE = "sumo_data.csv"

# --- Parameters for the Data Collection Policy (iot_control.py like) ---
CONGESTION_THRESHOLD_POLICY = 5 # vehicles
MIN_GREEN_TIME_IOT_POLICY = 10  # seconds
YELLOW_TIME_IOT_POLICY = 6      # seconds (as per your image)

# --- Mappings for 4 Main Green Phases ---
# Agent Action -> SUMO Green Phase Index
# 0: Green from S (SUMO Phase 0)
# 1: Green from E (SUMO Phase 2)
# 2: Green from N (SUMO Phase 4)
# 3: Green from W (SUMO Phase 6)
AGENT_ACTION_TO_SUMO_GREEN_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}
SUMO_GREEN_PHASE_TO_AGENT_ACTION = {v: k for k, v in AGENT_ACTION_TO_SUMO_GREEN_PHASE.items()}

# SUMO Green Phase Index -> SUMO Yellow Phase Index
SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7}

# --- State and Reward Definition for Logging (What the DQN will learn from) ---
# These define which SUMO edges correspond to "North", "South", "East", "West"
# for the *state representation* logged in the CSV and used by the DQN.
# Scenario: Event West, Commercial South & West, Commuter North. East is background.
APPROACH_EDGES_FOR_STATE_LOGGING = {
    "north": "754598165#2",   # Commuter
    "south": "1053267667#3",  # Commercial Hub
    "east": "749662140#0",    # Background/Other Commuter
    "west": "885403818#2",    # Event Venue & Other Commercial
}
VEHICLE_BINS_FOR_STATE_LOGGING = [5, 15, 30] # (0-4, 5-14, 15-29, 30+)

g_last_total_wait_time_log = 0
g_net_obj_policy_logger = None

def discretize_value_log(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold: return i
    return len(bins)

def get_sumo_state_for_log():
    try:
        n = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["north"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        s = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["south"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        e = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["east"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        w = discretize_value_log(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["west"]), VEHICLE_BINS_FOR_STATE_LOGGING)
        # current_tl_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID) # Optional: Add to state
        # agent_action_phase = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_phase_idx, -1) # Map SUMO phase to agent action
        # return (n, s, e, w, agent_action_phase) # If adding phase to state
        return (n, s, e, w)
    except Exception as e_state:
        print(f"Error in get_sumo_state_for_log: {e_state}")
        return None

def calculate_sumo_reward_for_log():
    global g_last_total_wait_time_log
    current_total = 0
    try:
        for edge_id in APPROACH_EDGES_FOR_STATE_LOGGING.values():
            current_total += traci.edge.getWaitingTime(edge_id)
        reward = (g_last_total_wait_time_log - current_total) / 100.0
        g_last_total_wait_time_log = current_total
        return reward
    except Exception as e_reward:
        print(f"Error in calculate_sumo_reward_for_log: {e_reward}")
        return 0

# --- Rule-Based Policy Logic (iot_control.py like) ---
# This dictionary maps SUMO edge IDs to named directions AS UNDERSTOOD BY THE RULE-BASED POLICY.
# **Ensure these edges and names align with your iot_control.py logic**
IOT_POLICY_EDGE_TO_DIRECTION_NAME = {
    "754598165#2": "north",  # Commuter
    "1053267667#3": "south", # Commercial
    "749662140#0": "east",   # Background
    "885403818#2": "west",   # Event
}

# Map named directions (as understood by policy) to the agent actions that serve them
# This is crucial for the rule-based policy to pick an action.
DIRECTION_TO_AGENT_ACTION_POLICY = {
    "south": 0, # Agent action 0 serves South (SUMO Phase 0)
    "east":  1, # Agent action 1 serves East  (SUMO Phase 2)
    "north": 2, # Agent action 2 serves North (SUMO Phase 4)
    "west":  3, # Agent action 3 serves West  (SUMO Phase 6)
}


def get_phase_green_directions_for_policy(tls_id, phase_state_str):
    global g_net_obj_policy_logger
    if g_net_obj_policy_logger is None: return set()
    green_directions = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        if not controlled_links: return green_directions
        for i, state_char in enumerate(phase_state_str):
            if state_char.lower() == 'g':
                if i < len(controlled_links) and controlled_links[i]:
                    first_connection_in_link = controlled_links[i][0]
                    if first_connection_in_link:
                        from_lane_id = first_connection_in_link[0]
                        from_lane_obj = g_net_obj_policy_logger.getLane(from_lane_id)
                        if from_lane_obj:
                            from_edge_id = from_lane_obj.getEdge().getID()
                            if from_edge_id in IOT_POLICY_EDGE_TO_DIRECTION_NAME:
                                green_directions.add(IOT_POLICY_EDGE_TO_DIRECTION_NAME[from_edge_id])
    except KeyError as e_key: print(f"Warning: Lane ID {e_key} not found.")
    except Exception as e_phase_dir: print(f"Warning: Error in get_phase_green_directions_for_policy: {e_phase_dir}")
    return green_directions


def get_rule_based_policy_action(current_tl_sumo_phase_idx):
    """
    Rule-based policy for 4 green phases.
    Prioritizes the most congested approach not currently being served.
    Returns an agent action (0, 1, 2, or 3) or None.
    """
    try:
        # Get vehicle counts/congestion for each named approach policy understands
        policy_approach_congestion = {} # direction_name: count
        for edge_id, direction_name in IOT_POLICY_EDGE_TO_DIRECTION_NAME.items():
            # Using HaltingNumber for simplicity, align with your iot_control.py if different
            count = traci.edge.getLastStepHaltingNumber(edge_id)
            policy_approach_congestion[direction_name] = count

        program_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
        current_phase_def = program_logic.phases[current_tl_sumo_phase_idx]
        current_phase_state_str = current_phase_def.state
        
        directions_served_by_current_green = get_phase_green_directions_for_policy(TRAFFIC_LIGHT_ID, current_phase_state_str)
        current_agent_action = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx)

        # Identify most congested approach *not* currently green (if current is green)
        max_congestion = -1
        most_congested_direction_to_serve = None

        # Sort approaches by congestion to find the highest priority
        # (direction_name, congestion_count)
        sorted_congestion = sorted(policy_approach_congestion.items(), key=lambda item: item[1], reverse=True)

        if 'g' in current_phase_state_str.lower(): # If current SUMO phase is green
            # Check if current green is serving a significantly congested direction
            is_current_serving_high_congestion = False
            for d in directions_served_by_current_green:
                if policy_approach_congestion.get(d, 0) >= CONGESTION_THRESHOLD_POLICY:
                    is_current_serving_high_congestion = True
                    break
            
            if is_current_serving_high_congestion and policy_approach_congestion.get(list(directions_served_by_current_green)[0],0) == sorted_congestion[0][1]:
                 # If current green serves the *most* congested direction (or one of them)
                 # and it's above threshold, policy might stay.
                return current_agent_action 

            # If not serving the most congested, or if current congestion is low, look for a switch.
            for direction_name, count in sorted_congestion:
                if count >= CONGESTION_THRESHOLD_POLICY:
                    if direction_name not in directions_served_by_current_green:
                        most_congested_direction_to_serve = direction_name
                        break # Found the most congested unserved direction
            
            if most_congested_direction_to_serve:
                return DIRECTION_TO_AGENT_ACTION_POLICY.get(most_congested_direction_to_serve)
            else: # No other direction is significantly congested, or current is fine
                return current_agent_action # Stay or no change if not green

        else: # Current is Yellow or Red
            # Find the most congested direction and switch to it
            for direction_name, count in sorted_congestion:
                if count >= CONGESTION_THRESHOLD_POLICY: # Simple threshold to trigger switch
                    return DIRECTION_TO_AGENT_ACTION_POLICY.get(direction_name)
            # If no significant congestion, maybe default to a base cycle (e.g., action 0)
            # or pick a default if nothing is congested (e.g. action for 'south' if that's a major route)
            return DIRECTION_TO_AGENT_ACTION_POLICY.get(sorted_congestion[0][0]) if sorted_congestion else 0


    except Exception as e_policy_action:
        print(f"Error in get_rule_based_policy_action: {e_policy_action}")
        traceback.print_exc()
        return random.choice(list(AGENT_ACTION_TO_SUMO_GREEN_PHASE.keys())) # Random on error

# --- Main Data Collection Function ---
def run_sumo_and_log_data(num_simulation_steps=30000, data_collection_policy_name="rule_based"):
    global g_last_total_wait_time_log, g_net_obj_policy_logger
    
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE]
    sumo_cmd_list.extend([
        "--waiting-time-memory", "1000", "--time-to-teleport", "-1",
        "--no-step-log", "true", "--seed", str(random.randint(0, 100000))
    ])

    traci.start(sumo_cmd_list)
    g_net_obj_policy_logger = net.readNet(NET_FILE)
    print(f"SUMO started for data collection with policy: {data_collection_policy_name}")

    # CSV Header: state_N, state_S, state_E, state_W, (optional: state_phase), action, reward, next_..., done
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
            
            for _ in range(5): traci.simulationStep(); current_step+=1 # Initialize
            
            state_s_for_log = get_sumo_state_for_log()
            action_applied_by_policy_for_log = -1

            while traci.simulation.getMinExpectedNumber() > 0 and current_step < num_simulation_steps:
                current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                is_currently_green = current_tl_sumo_phase_idx in AGENT_ACTION_TO_SUMO_GREEN_PHASE.values()
                
                if state_s_for_log is None: # Recapture state if lost
                    state_s_for_log = get_sumo_state_for_log()
                    if state_s_for_log is None:
                        traci.simulationStep(); current_step +=1; phase_decision_timer += traci.simulation.getDeltaT(); continue

                # Decision point
                if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME_IOT_POLICY:
                    if data_collection_policy_name == "rule_based":
                        decided_agent_action = get_rule_based_policy_action(current_tl_sumo_phase_idx)
                    elif data_collection_policy_name == "random":
                        decided_agent_action = random.choice(list(AGENT_ACTION_TO_SUMO_GREEN_PHASE.keys()))
                    else: # Default to random
                        decided_agent_action = random.choice(list(AGENT_ACTION_TO_SUMO_GREEN_PHASE.keys()))

                    if decided_agent_action is not None: # Policy made a decision
                        action_applied_by_policy_for_log = decided_agent_action
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
                        phase_decision_timer = 0.0 # Reset timer for the new/continued green phase
                    elif is_currently_green: # Policy decided to stay (decided_agent_action was None but it's green)
                         action_applied_by_policy_for_log = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx, -1)


                # Step SUMO
                traci.simulationStep()
                current_step += 1
                phase_decision_timer += traci.simulation.getDeltaT() # Always increment timer based on sim step
                
                # Observe results s', r, done
                state_s_prime_for_log = get_sumo_state_for_log()
                reward_for_log = calculate_sumo_reward_for_log()
                done_for_log = traci.simulation.getMinExpectedNumber() == 0 or current_step >= num_simulation_steps

                if state_s_for_log is not None and action_applied_by_policy_for_log != -1 and state_s_prime_for_log is not None:
                    row_to_write = list(state_s_for_log) + \
                                   [action_applied_by_policy_for_log, reward_for_log] + \
                                   list(state_s_prime_for_log) + \
                                   [done_for_log]
                    writer.writerow(row_to_write)
                    logged_transitions += 1
                
                state_s_for_log = state_s_prime_for_log
                action_applied_by_policy_for_log = -1 # Reset until next explicit decision

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
    run_sumo_and_log_data(num_simulation_steps=36000, data_collection_policy_name="rule_based") # ~10 hours of data if 1s steps
    # run_sumo_and_log_data(num_simulation_steps=10000, data_collection_policy_name="random")