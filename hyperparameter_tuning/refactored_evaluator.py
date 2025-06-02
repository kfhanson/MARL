import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import pandas as pd
import xml.etree.ElementTree as ET


# --- Default Hyperparameters (for model structure) ---
DEFAULT_STATE_FEATURES_EVAL = 4
DEFAULT_NUM_ACTIONS_EVAL = 4
DEFAULT_SEQUENCE_LENGTH_EVAL = 1

# --- SUMO & Control Params (should match what the model expects) ---
# SUMO_BINARY_EVAL = checkBinary('sumo') # Use 'sumo' for faster grid search eval
# CONFIG_FILE_EVAL = "osm.sumocfg"
# TRAFFIC_LIGHT_ID_EVAL = 'your_traffic_light_id'
# ACTION_TO_GREEN_PHASE_EVAL = {0: 0, 1: 2, 2: 4, 3: 6}
# GREEN_TO_YELLOW_PHASE_EVAL = {0: 1, 2: 3, 4: 5, 6: 7}
# MIN_GREEN_TIME_EVAL = 10
# YELLOW_TIME_EVAL = 6
# APPROACH_EDGES_FOR_STATE_EVAL = {...}
# VEHICLE_BINS_FOR_STATE_EVAL = [...]

class EvaluationAgent:
    def __init__(self, state_dims, action_size, sequence_length, model_path):
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.epsilon = 0.01 

        self.q_network = self._build_lstm_model()
        self.load_model(model_path)

    def _build_lstm_model(self):
        lstm_units = 32 
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(lstm_units, activation='relu', return_sequences=False),
            Dense(self.action_size, activation='linear')
        ])
        return model

    def load_model(self, filepath):
        try:
            self.q_network.load_weights(filepath)
        except Exception as e:
            print(f"Error loading evaluation model weights from {filepath}: {e}")
            raise

    def select_action(self, current_state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        if current_state is None: return np.random.randint(self.action_size)
        q_values = self.q_network.predict(current_state, verbose=0)
        return np.argmax(q_values[0])

def get_sumo_state_eval(approach_edges, vehicle_bins, seq_len, state_feat):
    try:
        n = discretize_value(traci.edge.getLastStepHaltingNumber(approach_edges["north"]), vehicle_bins)
        s = discretize_value(traci.edge.getLastStepHaltingNumber(approach_edges["south"]), vehicle_bins)
        e = discretize_value(traci.edge.getLastStepHaltingNumber(approach_edges["east"]), vehicle_bins)
        w = discretize_value(traci.edge.getLastStepHaltingNumber(approach_edges["west"]), vehicle_bins)
        state_vector = np.array([n, s, e, w], dtype=np.float32)
        return state_vector.reshape((1, seq_len, state_feat))
    except Exception: return None

def discretize_value(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold: return i
    return len(bins)

def evaluate_model_in_sumo(
    model_load_path,
    sumo_binary_path, config_file_path, net_file_path, traffic_light_id,
    action_to_green_phase_map, green_to_yellow_phase_map,
    min_green_time_val, yellow_time_val,
    approach_edges_map, vehicle_bins_list,
    state_features_val=DEFAULT_STATE_FEATURES_EVAL,
    num_actions_val=DEFAULT_NUM_ACTIONS_EVAL,
    sequence_length_val=DEFAULT_SEQUENCE_LENGTH_EVAL,
    num_eval_episodes=3, max_steps_eval=3600, base_seed=10000
):
    print(f"Evaluating model: {model_load_path}")
    agent = EvaluationAgent(state_features_val, num_actions_val, sequence_length_val, model_load_path)
    
    all_episode_avg_wait_times = []

    for i in range(num_eval_episodes):
        episode_num_eval = i + 1
        tripinfo_filename = f"tripinfo_eval_temp_trial_ep{episode_num_eval}.xml" # Temporary
        
        sumo_cmd_list = [sumo_binary_path, "-c", config_file_path]
        sumo_cmd_list.extend([
            "--tripinfo-output", tripinfo_filename,
            "--waiting-time-memory", "1000", "--time-to-teleport", "-1",
            "--no-step-log", "true", "--seed", str(base_seed + i)
        ])

        traci.start(sumo_cmd_list)
        current_step = 0
        phase_decision_timer = 0.0
        current_sumo_state = get_sumo_state_eval(approach_edges_map, vehicle_bins_list, sequence_length_val, state_features_val)

        if current_sumo_state is None:
            print("  Error: Failed to get initial SUMO state. Skipping eval episode.")
            traci.close()
            continue
        
        try:
            while traci.simulation.getMinExpectedNumber() > 0 and current_step < max_steps_eval:
                current_tl_phase = traci.trafficlight.getPhase(traffic_light_id)
                is_green = current_tl_phase in action_to_green_phase_map.values()

                if not is_green or phase_decision_timer >= min_green_time_val:
                    action = agent.select_action(current_sumo_state)
                    target_phase = action_to_green_phase_map[action]
                    if current_tl_phase != target_phase:
                        if current_tl_phase in green_to_yellow_phase_map:
                            yellow_phase = green_to_yellow_phase_map[current_tl_phase]
                            traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
                            for _ in range(int(yellow_time_val / traci.simulation.getDeltaT())):
                                if traci.simulation.getMinExpectedNumber() == 0: break
                                traci.simulationStep(); current_step +=1
                            if traci.simulation.getMinExpectedNumber() == 0: break
                        traci.trafficlight.setPhase(traffic_light_id, target_phase)
                    phase_decision_timer = 0.0
                
                traci.simulationStep()
                current_step += 1
                phase_decision_timer += traci.simulation.getDeltaT()
                current_sumo_state = get_sumo_state_eval(approach_edges_map, vehicle_bins_list, sequence_length_val, state_features_val)
                if current_sumo_state is None and traci.simulation.getMinExpectedNumber() > 0: break
                if traci.simulation.getMinExpectedNumber() == 0: break
        finally:
            if 'traci' in sys.modules and traci.isLoaded() and traci.getConnection(): traci.close()

        avg_wait = 0
        if os.path.exists(tripinfo_filename):
            tree = ET.parse(tripinfo_filename)
            root = tree.getroot()
            wait_times = [float(trip.get('waitingTime')) for trip in root.findall('tripinfo') if trip.get('arrival') != '-1']
            if wait_times: avg_wait = np.mean(wait_times)
            all_episode_avg_wait_times.append(avg_wait)
            os.remove(tripinfo_filename)
    
    return np.mean(all_episode_avg_wait_times) if all_episode_avg_wait_times else float('inf')