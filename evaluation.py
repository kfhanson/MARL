import os
import sys
import traci
import numpy as np
import random
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam #
import traceback

try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")

SUMO_BINARY = checkBinary('sumo')
CONFIG_FILE = "osm.sumocfg"
TRAFFIC_LIGHT_ID = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'
MODEL_LOAD_PATH = "offline_dqn_agent.weights.h5"

# --- Agent/Environment Parameters (MUST MATCH TRAINING SETUP) ---
STATE_FEATURES = 4  # N, S, E, W discretized congestion counts
NUM_ACTIONS = 4     # Serve S (0), E (1), N (2), W (3)
SEQUENCE_LENGTH = 1 # Using current state as a sequence of length 1

EVAL_EPSILON = 0.05

# --- Traffic Light Control Parameters (MUST MATCH TRAINING/DATA COLLECTION) ---
MIN_GREEN_TIME = 10  # seconds
YELLOW_TIME = 6      # seconds
ACTION_TO_GREEN_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}  # Agent action to SUMO Green Phase Index
GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7} # SUMO Green Phase to its Yellow Phase

# --- State Definition (MUST MATCH TRAINING/DATA COLLECTION) ---
APPROACH_EDGES_FOR_STATE = {
    "north": "754598165#2",
    "south": "1053267667#3",
    "east": "749662140#0",
    "west": "885403818#2",
}
VEHICLE_BINS_FOR_STATE = [5, 15, 30]

def discretize_value(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold: return i
    return len(bins)

def get_sumo_state_eval():
    """ Retrieves current state from SUMO for evaluation. """
    try:
        n = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["north"]), VEHICLE_BINS_FOR_STATE)
        s = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["south"]), VEHICLE_BINS_FOR_STATE)
        e = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["east"]), VEHICLE_BINS_FOR_STATE)
        w = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["west"]), VEHICLE_BINS_FOR_STATE)
        state_vector = np.array([n, s, e, w], dtype=np.float32)
        return state_vector.reshape((1, SEQUENCE_LENGTH, STATE_FEATURES))
    except Exception as e_state:
        print(f"Error in get_sumo_state_eval: {e_state}")
        return None

class EvaluationAgent:
    def __init__(self, state_dims, action_size, sequence_length, model_path):
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.epsilon = EVAL_EPSILON

        self.q_network = self._build_lstm_model()
        self.load_model(model_path)
        print("Evaluation Agent Initialized and Model Loaded.")
        self.q_network.summary()


    def _build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(32, activation='relu', return_sequences=False),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def load_model(self, filepath):
        try:
            self.q_network.load_weights(filepath)
            print(f"Evaluation model weights loaded from {filepath}")
        except Exception as e:
            print(f"Error loading evaluation model weights: {e}")
            raise

    def select_action(self, current_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if current_state is None:
             print("Warning: current_state is None in select_action, choosing random action.")
             return random.randrange(self.action_size)
        q_values = self.q_network.predict(current_state, verbose=0)
        return np.argmax(q_values[0])


def run_evaluation_episode(agent, episode_num_eval, max_steps_eval=3600, sim_seed_offset=0):
    """Runs a single SUMO episode with the trained agent for evaluation."""
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE]
    sumo_cmd_list.extend([
        "--tripinfo-output", f"tripinfo_eval_offline_4act_ep{episode_num_eval}.xml",
        "--summary-output", f"summary_eval_offline_4act_ep{episode_num_eval}.xml",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--no-step-log", "true",
        "--seed", str(random.randint(10000, 20000) + sim_seed_offset)
    ])
    if SUMO_BINARY.endswith("-gui"): sumo_cmd_list.append("--step-length=0.2")

    traci.start(sumo_cmd_list)
    print(f"  SUMO evaluation episode {episode_num_eval} started (Epsilon: {agent.epsilon}).")

    current_step = 0
    phase_decision_timer = 0.0
    
    current_sumo_state_for_eval = get_sumo_state_eval()
    if current_sumo_state_for_eval is None:
        print("  Error: Failed to get initial SUMO state for evaluation. Ending episode.")
        traci.close()
        return

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and current_step < max_steps_eval:
            current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            is_currently_green = current_tl_sumo_phase_idx in ACTION_TO_GREEN_PHASE.values()

            if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME:
                agent_action_choice = agent.select_action(current_sumo_state_for_eval)
                target_sumo_green_phase = ACTION_TO_GREEN_PHASE[agent_action_choice]

                if current_tl_sumo_phase_idx != target_sumo_green_phase:
                    if current_tl_sumo_phase_idx in GREEN_TO_YELLOW_PHASE:
                        yellow_phase_idx = GREEN_TO_YELLOW_PHASE[current_tl_sumo_phase_idx]
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                        for _ in range(int(YELLOW_TIME / traci.simulation.getDeltaT())):
                            if traci.simulation.getMinExpectedNumber() == 0: break
                            traci.simulationStep(); current_step += 1
                        if traci.simulation.getMinExpectedNumber() == 0: break
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                    else:
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                    phase_decision_timer = 0.0
                # If agent chooses to stay in current green, timer continues.
            
            traci.simulationStep()
            current_step += 1
            phase_decision_timer += traci.simulation.getDeltaT()

            current_sumo_state_for_eval = get_sumo_state_eval()
            if current_sumo_state_for_eval is None and traci.simulation.getMinExpectedNumber() > 0:
                print("  Error: Lost SUMO state mid-evaluation. Ending episode.")
                break
            
            if traci.simulation.getMinExpectedNumber() == 0: break
        
        print(f"  Evaluation episode {episode_num_eval} finished. Steps: {current_step}.")

    except traci.exceptions.FatalTraCIError as e: print(f"Fatal TraCI Error during evaluation: {e}")
    except KeyboardInterrupt: print("Evaluation interrupted by user.")
    except Exception as e:
        print(f"Unexpected Python error during evaluation episode:")
        traceback.print_exc()
    finally:
        try:
            if 'traci' in sys.modules:
                traci.close()
                print(f"  TraCI connection closed for episode {episode_num_eval}.")
        except traci.exceptions.TraCIException as e:
            print(f"  TraCI warning on close for episode {episode_num_eval}: {e}")
        except Exception as e_close:
            print(f"  Unexpected error during TraCI close for episode {episode_num_eval}: {e_close}")


if __name__ == "__main__":
    print("Starting Evaluation of Offline Trained DQN Agent (4 Actions)...")
    
    # Create the agent instance and load the trained weights
    evaluation_agent = EvaluationAgent(STATE_FEATURES, NUM_ACTIONS, SEQUENCE_LENGTH, MODEL_LOAD_PATH)
    num_eval_episodes = 5 
    for i in range(num_eval_episodes):
        run_evaluation_episode(evaluation_agent, i + 1, sim_seed_offset=i*100)
    
    print("\nEvaluation complete.")