import os
import sys
import traci
import time
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
# from sumolib import net, checkBinary # If needed for network info
import traceback

# --- SUMO Configuration (same as before) ---
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary
    else:
        sys.exit("SUMO_HOME not found!")
except ImportError as e:
    sys.exit(f"Error importing SUMO tools: {e}")

sumo_binary = checkBinary('sumo-gui')
sumo_config = "osm.sumocfg"
# net_file = "osm.net.xml" # May not be directly needed in agent code unless for phase counts

traffic_light_id = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'

# --- DQN Parameters ---
STATE_SIZE = 4 # Number of features in our state (e.g., discretized counts for N, S, E, W)
               # If you add current_phase, this becomes 5. Adjust accordingly.
ACTION_SIZE = 2  # Number of possible actions (e.g., switch to N/S green, switch to E/W green)

LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999 # Decay epsilon per episode or per fixed number of steps
BATCH_SIZE = 32 # Size of mini-batch from replay buffer
REPLAY_BUFFER_SIZE = 10000 # Max size of replay buffer
TARGET_UPDATE_FREQUENCY = 100 # Update target network every C steps/episodes

# --- Traffic Light Control Parameters ---
MIN_GREEN_TIME = 10
YELLOW_TIME = 3
action_to_green_phase = {0: 0, 1: 2} # N/S Green, E/W Green
green_to_yellow_phase = {0: 1, 2: 3} # Corresponding Yellow phases

# --- State Discretization & Definition ---
# Using vehicle counts as per your previous sumo_qlearning.py
vehicle_bins = [5, 15, 30] # Discretized levels: 0, 1, 2, 3
approach_edges = {
    "north": "754598165#2",
    "south": "1053267667#3",
    "east": "749662140#0",
    "west": "885403818#2",
}

def discretize(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold:
            return i
    return len(bins)

last_total_wait_time = 0 # For reward calculation

def get_state_from_traci(tls_id):
    try:
        # Using halting numbers for now. Consider occupancy or normalized queues.
        north_count = traci.edge.getLastStepHaltingNumber(approach_edges["north"])
        south_count = traci.edge.getLastStepHaltingNumber(approach_edges["south"])
        east_count  = traci.edge.getLastStepHaltingNumber(approach_edges["east"])
        west_count  = traci.edge.getLastStepHaltingNumber(approach_edges["west"])

        # current_phase = traci.trafficlight.getPhase(tls_id) # Optional: add to state

        # State features: normalized or discretized values
        state_features = np.array([
            discretize(north_count, vehicle_bins),
            discretize(south_count, vehicle_bins),
            discretize(east_count, vehicle_bins),
            discretize(west_count, vehicle_bins),
            # float(current_phase) / MAX_PHASE_INDEX # If current_phase is added & normalized
        ])
        # For LSTM, input shape is (batch_size, timesteps, features)
        # If using a single timestep, reshape to (1, 1, STATE_SIZE)
        return state_features.reshape((1, 1, STATE_SIZE)) # sequence_length=1
    except traci.exceptions.TraCIException as e:
        print(f"TraCI error getting state: {e}. Returning None.")
        return None
    except Exception as e:
        print(f"Unexpected error getting state: {e}. Returning None.")
        traceback.print_exc()
        return None


def calculate_reward(tls_id):
    global last_total_wait_time
    try:
        current_total_wait_time = 0
        for edge_id in approach_edges.values():
            current_total_wait_time += traci.edge.getWaitingTime(edge_id)
        
        reward = (last_total_wait_time - current_total_wait_time) / 100.0 # Normalize reward slightly
        # reward = -current_total_wait_time / 100.0 # Simpler negative reward

        last_total_wait_time = current_total_wait_time
        return reward
    except traci.exceptions.TraCIException:
        return 0 # Default reward on error

# --- DQN Agent Class ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.learning_rate = LEARNING_RATE

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Initialize target model weights

    def _build_model(self):
        # LSTM expects input_shape=(timesteps, features)
        # For a single state observation at a time, timesteps = 1
        model = Sequential()
        model.add(Input(shape=(1, self.state_size))) # Define input shape explicitly for clarity
        model.add(LSTM(24, activation='relu')) # Number of LSTM units can be tuned
        # model.add(Dense(24, activation='relu')) # Optional additional Dense layer
        model.add(Dense(self.action_size, activation='linear')) # Output Q-values for each action
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        print("Model Summary:")
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        q_values = self.model.predict(state, verbose=0) # state should be (1, 1, state_size)
        return np.argmax(q_values[0]) # Exploit

    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return # Not enough samples to replay

        minibatch = random.sample(self.replay_buffer, batch_size)

        states = np.array([i[0][0] for i in minibatch]) # Extract features, original shape (N, 1, state_size) -> (N, state_size) for LSTM
        states = np.reshape(states, [batch_size, 1, self.state_size]) # Reshape for LSTM batch input

        next_states = np.array([i[3][0] for i in minibatch])
        next_states = np.reshape(next_states, [batch_size, 1, self.state_size])

        # Get current Q-values for the states in the batch
        current_q_values_batch = self.model.predict(states, verbose=0)
        # Get target Q-values for the next_states in the batch
        target_q_values_next_batch = self.target_model.predict(next_states, verbose=0)


        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(target_q_values_next_batch[i])
            
            current_q_values = current_q_values_batch[i]
            current_q_values[action] = target
            # current_q_values_batch[i] is already updated by reference here

        # Train the model on the updated Q-values (targets)
        self.model.fit(states, current_q_values_batch, epochs=1, verbose=0)


    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)

# --- Main Simulation Loop ---
def run_simulation_dqn(num_episodes=500): # Number of simulation runs for training
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    # agent.load("traffic_dqn.weights.h5") # Uncomment to load pre-trained weights

    for e in range(num_episodes):
        sumo_cmd = [
            sumo_binary, "-c", sumo_config,
            "--tripinfo-output", f"tripinfo_dqn_ep{e}.xml",
            # "--summary-output", f"summary_dqn_ep{e}.xml", # Optional
            "--waiting-time-memory", "1000",
            "--duration-log.statistics",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
            # "--seed", str(random.randint(0, 10000)) # Vary seed per episode for diverse experience
        ]
        if sumo_binary.endswith("-gui"): # If using GUI, don't run too fast initially
             sumo_cmd += ["--step-length", "0.1"] # for better visualization, remove for faster training

        traci.start(sumo_cmd)
        print(f"Episode {e+1}/{num_episodes} started. Epsilon: {agent.epsilon:.4f}")

        step = 0
        simulation_duration_steps = 3600 # Max steps per episode (e.g., 1 hour at 1s/step)
        phase_start_time = 0
        global last_total_wait_time
        last_total_wait_time = 0 # Reset for each episode

        state = get_state_from_traci(traffic_light_id)
        if state is None:
            print("Error: Could not get initial state. Ending episode.")
            traci.close()
            continue

        total_reward_episode = 0
        update_target_counter = 0

        try:
            while traci.simulation.getMinExpectedNumber() > 0 and step < simulation_duration_steps:
                current_time = traci.simulation.getTime()
                current_sumo_phase = traci.trafficlight.getPhase(traffic_light_id)

                is_green_phase_active = current_sumo_phase in action_to_green_phase.values()
                min_time_passed = (current_time - phase_start_time) >= MIN_GREEN_TIME

                action_taken_this_step = False

                if (is_green_phase_active and min_time_passed) or not is_green_phase_active:
                    chosen_action = agent.choose_action(state) # state should be (1, 1, state_size)
                    target_green_phase = action_to_green_phase[chosen_action]
                    action_taken_this_step = True

                    if current_sumo_phase != target_green_phase:
                        if current_sumo_phase in green_to_yellow_phase:
                            yellow_phase = green_to_yellow_phase[current_sumo_phase]
                            traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
                            yellow_end_time = current_time + YELLOW_TIME
                            while traci.simulation.getTime() < yellow_end_time:
                                if traci.simulation.getMinExpectedNumber() == 0: break
                                traci.simulationStep(); step += 1 # Advance SUMO
                            
                            if traci.simulation.getMinExpectedNumber() == 0: break
                            traci.trafficlight.setPhase(traffic_light_id, target_green_phase)
                            phase_start_time = traci.simulation.getTime()
                        else:
                            traci.trafficlight.setPhase(traffic_light_id, target_green_phase)
                            phase_start_time = traci.simulation.getTime()
                    # If already in target_green_phase and min_time has passed,
                    # agent might choose same action, effectively extending it until next decision.

                # Simulate one SUMO step AFTER action decision and potential phase change
                traci.simulationStep()
                step += 1

                next_state = get_state_from_traci(traffic_light_id)
                reward = calculate_reward(traffic_light_id) if action_taken_this_step else 0 # Only give reward if an action was decided
                total_reward_episode += reward
                done = traci.simulation.getMinExpectedNumber() == 0 or step >= simulation_duration_steps

                if next_state is not None and action_taken_this_step: # Only remember if an action was taken
                    agent.remember(state, chosen_action, reward, next_state, done)
                
                state = next_state
                if state is None and not done: # Handle rare case where next state is None mid-episode
                    print("Error: Lost state mid-episode. Breaking.")
                    break

                if len(agent.replay_buffer) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
                
                update_target_counter += 1
                if update_target_counter % TARGET_UPDATE_FREQUENCY == 0:
                    agent.update_target_model()
                    print(f"    Target network updated at step {step}")

                if done:
                    break
            
            agent.decay_epsilon() # Decay epsilon at the end of each episode
            print(f"Episode {e+1} finished after {step} steps. Total Reward: {total_reward_episode:.2f}")

        except traci.exceptions.FatalTraCIError as err:
            print(f"Fatal TraCI Error in episode {e+1}: {err}")
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            agent.save("traffic_dqn_interrupted.weights.h5")
            traci.close()
            return
        except Exception as err:
            print(f"Unexpected Python error in episode {e+1} at step {step}:")
            traceback.print_exc()
        finally:
            if 'traci' in sys.modules and traci.isEmbedded(): # Check if traci is connected
                traci.close()
            print("-" * 30)

    agent.save("traffic_dqn_final.weights.h5")
    print("Training finished and model saved.")


if __name__ == "__main__":
    # Ensure script is run from a directory where it can find osm.sumocfg
    # os.chdir(path_to_your_sumo_scenario_directory) # If needed
    run_simulation_dqn(num_episodes=100) # Start with fewer episodes for testing