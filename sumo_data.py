import os
import sys
import traci
import csv
import numpy as np # For potential state formatting if needed later
import random
import traceback

# --- SUMO Configuration ---
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")

SUMO_BINARY = checkBinary('sumo') # Use 'sumo' for faster data collection
CONFIG_FILE = "osm.sumocfg"
TRAFFIC_LIGHT_ID = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'
OUTPUT_CSV_FILE = "sumo_offline_rl_data.csv"

# --- Traffic Light Control Parameters (for data collection policy) ---
MIN_GREEN_TIME_POLICY = 10
YELLOW_TIME_POLICY = 3
ACTION_TO_GREEN_PHASE_POLICY = {0: 0, 1: 2}
GREEN_TO_YELLOW_PHASE_POLICY = {0: 1, 2: 3}

# --- State Definition (consistent with what the DQN will use) ---
APPROACH_EDGES_POLICY = {
    "north": "754598165#2",
    "south": "1053267667#3",
    "east": "749662140#0",
    "west": "885403818#2",
}
VEHICLE_BINS_POLICY = [5, 15, 30]

def discretize_value_policy(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold:
            return i
    return len(bins)

g_last_total_wait_time_policy = 0

def get_sumo_state_policy():
    try:
        n_halt = discretize_value_policy(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_POLICY["north"]), VEHICLE_BINS_POLICY)
        s_halt = discretize_value_policy(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_POLICY["south"]), VEHICLE_BINS_POLICY)
        e_halt = discretize_value_policy(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_POLICY["east"]), VEHICLE_BINS_POLICY)
        w_halt = discretize_value_policy(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_POLICY["west"]), VEHICLE_BINS_POLICY)
        # current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID) # Optional: add to logged state
        return (n_halt, s_halt, e_halt, w_halt) # Log as a tuple of numbers
    except Exception: return None

def calculate_sumo_reward_policy():
    global g_last_total_wait_time_policy
    current_total_wait_time = 0
    try:
        for edge_id in APPROACH_EDGES_POLICY.values():
            current_total_wait_time += traci.edge.getWaitingTime(edge_id)
        reward = (g_last_total_wait_time_policy - current_total_wait_time) / 100.0
        g_last_total_wait_time_policy = current_total_wait_time
        return reward
    except Exception: return 0


def run_sumo_and_log_data(num_simulation_steps=10000, data_collection_policy="random"):
    """
    Runs SUMO with a specified policy and logs (s, a, r, s', done) transitions.
    :param num_simulation_steps: Total SUMO steps to run for data collection.
    :param data_collection_policy: "random", "fixed_alternating", or implement your own.
    """
    global g_last_total_wait_time_policy
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE]
    sumo_cmd_list.extend([
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--no-step-log", "true",
        "--seed", str(random.randint(0, 100000)) # Vary data collection runs
    ])

    traci.start(sumo_cmd_list)
    print(f"SUMO started for data collection with policy: {data_collection_policy}")

    # CSV Header: state_N, state_S, state_E, state_W, action, reward, next_state_N, next_state_S, next_state_E, next_state_W, done
    csv_header = [
        "state_N", "state_S", "state_E", "state_W", "action", "reward",
        "next_state_N", "next_state_S", "next_state_E", "next_state_W", "done"
    ]
    # If you add current_phase to state: "state_phase", "next_state_phase"

    logged_transitions = 0
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            current_step = 0
            g_last_total_wait_time_policy = 0
            phase_decision_timer = 0
            last_state_tuple = get_sumo_state_policy()
            last_agent_action = -1 # The action that led to the current last_state_tuple

            # Policy specific variables
            alternating_action_counter = 0

            while traci.simulation.getMinExpectedNumber() > 0 and current_step < num_simulation_steps:
                sim_time = traci.simulation.getTime()
                current_tl_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                is_currently_green = current_tl_phase_idx in ACTION_TO_GREEN_PHASE_POLICY.values()

                chosen_agent_action_for_log = -1 # This is the action we decide to take NOW

                if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME_POLICY:
                    # --- Data Collection Policy ---
                    if data_collection_policy == "random":
                        chosen_agent_action_for_log = random.choice([0, 1])
                    elif data_collection_policy == "fixed_alternating":
                        chosen_agent_action_for_log = alternating_action_counter % 2
                        alternating_action_counter += 1
                    # Add more policies here (e.g., your previous rule-based one)
                    else: # Default to random if policy unknown
                        chosen_agent_action_for_log = random.choice([0, 1])
                    
                    target_sumo_green_phase = ACTION_TO_GREEN_PHASE_POLICY[chosen_agent_action_for_log]

                    if current_tl_phase_idx != target_sumo_green_phase:
                        if current_tl_phase_idx in GREEN_TO_YELLOW_PHASE_POLICY:
                            yellow_phase_idx = GREEN_TO_YELLOW_PHASE_POLICY[current_tl_phase_idx]
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                            for _ in range(int(YELLOW_TIME_POLICY / traci.simulation.getDeltaT())):
                                if traci.simulation.getMinExpectedNumber() == 0: break
                                traci.simulationStep(); current_step += 1
                            if traci.simulation.getMinExpectedNumber() == 0: break
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                        else:
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                        phase_decision_timer = 0 # Reset timer after phase change
                
                # --- Simulate one SUMO step ---
                traci.simulationStep()
                current_step += 1
                if is_currently_green: # If it *was* green, increment its timer
                    phase_decision_timer += traci.simulation.getDeltaT()


                # --- Observe results and Log ---
                current_state_tuple = get_sumo_state_policy() # This is s'
                reward = calculate_sumo_reward_policy()      # This is r for transition (s,a) -> s'
                done = traci.simulation.getMinExpectedNumber() == 0 or current_step >= num_simulation_steps

                # Log the transition (last_state_tuple (s), last_agent_action (a), reward (r), current_state_tuple (s'), done)
                if last_state_tuple is not None and last_agent_action != -1 and current_state_tuple is not None:
                    row_to_write = list(last_state_tuple) + \
                                   [last_agent_action, reward] + \
                                   list(current_state_tuple) + \
                                   [done]
                    writer.writerow(row_to_write)
                    logged_transitions += 1
                
                last_state_tuple = current_state_tuple
                last_agent_action = chosen_agent_action_for_log # The action taken that will lead to the *next* state

                if done: break
            
            print(f"Data collection finished. Logged {logged_transitions} transitions to {OUTPUT_CSV_FILE}.")

    except traci.exceptions.FatalTraCIError as e:
        print(f"Fatal TraCI Error during data collection: {e}")
    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"Unexpected Python error during data collection:")
        traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isEmbedded():
            traci.close()

if __name__ == "__main__":
    # Run data collection for a certain number of steps
    # Choose a policy: "random", "fixed_alternating"
    run_sumo_and_log_data(num_simulation_steps=20000, data_collection_policy="random")
    # run_sumo_and_log_data(num_simulation_steps=20000, data_collection_policy="fixed_alternating")

    # You might want to run multiple data collection runs with different policies
    # and seeds, then concatenate the CSV files (ensuring headers are handled).