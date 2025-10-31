import os
import sys
import traci
import numpy as np
import random
import time
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from sumolib import checkBinary

SUMO_BINARY = "sumo-gui"
CORRIDOR_CONFIG_FILE = "bsd.sumocfg"
TRAFFIC_LIGHT_IDS = ["cluster_12705056632_3639980474_3640024452_3640024453_#7more", "cluster_3640024470_3640024471_3640024476_699593339_#8more"]
LOCAL_MODEL_FILE = "marl_dqn_local.weights.h5" 

STATE_FEATURES = 6
NUM_ACTIONS = 4
SEQUENCE_LENGTH = 1

EPISODES = 100
MAX_STEPS_PER_EPISODE = 3600
GAMMA = 0.95
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 50000  
TRAINING_START_BUFFER_SIZE = 1000 
TRAINING_INTERVAL = 4 
TARGET_UPDATE_FREQ = 200

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
g_epsilon = EPSILON_START

NEIGHBOR_EDGE_MAP = {
    "cluster_12705056632_3639980474_3640024452_3640024453_#7more": {
        "neighbor_incoming_edge": ""
    },
    "cluster_3640024470_3640024471_3640024476_699593339_#8more": {
        "neighbor_incoming_edge": ""
    }
}

# Traffic Light Control Parameters
MIN_GREEN_TIME = 10
YELLOW_TIME = 4
ACTION_TO_GREEN_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}
GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7}

# State Definition
LOCAL_APPROACH_EDGES = {
    "cluster_12705056632_3639980474_3640024452_3640024453_#7more": {
        "north": "",
        "south": "",
        "east": "",
        "west": "",
    },
    "cluster_3640024470_3640024471_3640024476_699593339_#8more": {
        "north": "",
        "south": "",
        "east": "",
        "west": "",
    }
}
VEHICLE_BINS_FOR_STATE = [5, 15, 30]

def discretize_value(value, bins):
    for i, threshold in enumerate(bins):
        if value <= threshold:
            return i
    return len(bins)

def get_multi_agent_sumo_state(tls_id):
    try:
        #Get local state for each traffic light
        local_edges = LOCAL_APPROACH_EDGES[tls_id]
        n = discretize_value(traci.edge.getLastStepHaltingNumber(local_edges["north"]), VEHICLE_BINS_FOR_STATE)
        s = discretize_value(traci.edge.getLastStepHaltingNumber(local_edges["south"]), VEHICLE_BINS_FOR_STATE)
        e = discretize_value(traci.edge.getLastStepHaltingNumber(local_edges["east"]), VEHICLE_BINS_FOR_STATE)
        w = discretize_value(traci.edge.getLastStepHaltingNumber(local_edges["west"]), VEHICLE_BINS_FOR_STATE)

        neighbor_info = NEIGHBOR_EDGE_MAP.get(tls_id, {})
        neighbor_queues = []
        neighbor_edge = neighbor_info.get("neighbor_incoming_edge")
        neighbor_q = discretize_value(traci.edge.getLastStepHaltingNumber(neighbor_edge), VEHICLE_BINS_FOR_STATE) if neighbor_edge else 0

        state_vector = np.array([n, s, e, w, neighbor_q], dtype=np.float32)
        return state_vector.reshape((1, SEQUENCE_LENGTH, STATE_FEATURES))
    except traci.TraCIException as e:
        return None

def calculate_hybrid_reward(tls_id, all_tls_ids):
    try:
        # Local reward
        local_wait_time = sum(traci.edge.getWaitingTime(edge) for edge in LOCAL_APPROACH_EDGES[tls_id].values())
        local_reward = -local_wait_time

        # Global reward
        total_corridor_wait_time = 0
        num_edges = 0
        for tid in all_tls_ids:
            edges = LOCAL_APPROACH_EDGES[tid].values()
            total_corridor_wait_time += sum(traci.edge.getWaitingTime(edge) for edge in edges)
            num_edges += len(edges)

        global_reward = - (total_corridor_wait_time / num_edges) if num_edges > 0 else 0

        hybrid_reward = (0.7 * local_reward) + (0.3 * global_reward)
        return hybrid_reward
    except traci.TraCIException as e:
        print(f"TraCI Exception in reward calculation for {tls_id}: {e}")
        return 0

class DQNAgent:
    def __init__(self, state_dims, action_size, sequence_length, learning_rate, gamma):
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_network = self._build_lstm_model()
        self.target_network = self._build_lstm_model()
        self.update_target_network()

    def _build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def train_on_batch(self, batch_experiences):
        current_states = np.array([exp[0] for exp in batch_experiences]).reshape(-1, self.sequence_length, self.state_feature_size)
        next_states = np.array([exp[3] for exp in batch_experiences]).reshape(-1, self.sequence_length, self.state_feature_size)
        
        q_current = self.q_network.predict(current_states, verbose=0)
        q_next_target = self.target_network.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(batch_experiences):
            if done:
                q_current[i][action] = reward
            else:
                q_current[i][action] = reward + self.gamma * np.amax(q_next_target[i])
        
        self.q_network.fit(current_states, q_current, batch_size=len(batch_experiences), epochs=1, verbose=0)

    def select_action(self, current_state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(current_state, verbose=0)
        return np.argmax(q_values[0])


    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.q_network.load_weights(filepath)
            self.update_target_network()
            print(f"Local model loaded from {filepath}")
    
    def save_model(self, filepath):
        self.q_network.save_weights(filepath)


if __name__ == "__main__":
    # --- 1. Initialize Agents and Replay Buffers ---
    agents = {}
    replay_buffers = {}
    for tls_id in TRAFFIC_LIGHT_IDS:
        agents[tls_id] = DQNAgent(STATE_FEATURES, NUM_ACTIONS, SEQUENCE_LENGTH, LEARNING_RATE, GAMMA)
        replay_buffers[tls_id] = deque(maxlen=MEMORY_SIZE)

    agents[TRAFFIC_LIGHT_IDS[0]].load_model(LOCAL_MODEL_FILE)
    for i in range(1, len(TRAFFIC_LIGHT_IDS)):
        agents[TRAFFIC_LIGHT_IDS[i]].q_network.set_weights(agents[TRAFFIC_LIGHT_IDS[0]].q_network.get_weights())
        agents[TRAFFIC_LIGHT_IDS[i]].update_target_network()

    # --- 2. Main Training Loop ---
    total_steps = 0
    for episode in range(EPISODES):
        sumo_cmd = [checkBinary(SUMO_BINARY), "-c", CORRIDOR_CONFIG_FILE, "--time-to-teleport", "-1", "--waiting-time-memory", "1000", "--random", "true"]
        traci.start(sumo_cmd)
        
        step = 0
        episode_rewards = {tls_id: 0 for tls_id in TRAFFIC_LIGHT_IDS}
        
        last_states = {tls_id: None for tls_id in TRAFFIC_LIGHT_IDS}
        last_actions = {tls_id: None for tls_id in TRAFFIC_LIGHT_IDS}
        phase_timers = {tls_id: 0.0 for tls_id in TRAFFIC_LIGHT_IDS}

        while step < MAX_STEPS_PER_EPISODE and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            for tls_id in TRAFFIC_LIGHT_IDS:
                current_phase = traci.trafficlight.getPhase(tls_id)
                is_green = current_phase in ACTION_TO_GREEN_PHASE.values()
                
                if not is_green or phase_timers[tls_id] >= MIN_GREEN_TIME:
                    current_state = get_multi_agent_sumo_state(tls_id)
                    if current_state is None: continue

                    if last_states[tls_id] is not None:
                        reward = calculate_hybrid_reward(tls_id, TRAFFIC_LIGHT_IDS)
                        done = traci.simulation.getMinExpectedNumber() == 0
                        
                        replay_buffers[tls_id].append((
                            last_states[tls_id].flatten(),
                            last_actions[tls_id],
                            reward,
                            current_state.flatten(),
                            done
                        ))
                        episode_rewards[tls_id] += reward

                    action = agents[tls_id].select_action(current_state, g_epsilon)
                    
                    last_states[tls_id] = current_state
                    last_actions[tls_id] = action

                    target_green_phase = ACTION_TO_GREEN_PHASE[action]
                    if current_phase != target_green_phase:
                        if current_phase in GREEN_TO_YELLOW_PHASE:
                            yellow_phase = GREEN_TO_YELLOW_PHASE[current_phase]
                            traci.trafficlight.setPhase(tls_id, yellow_phase)
                            yellow_duration = int(YELLOW_TIME / traci.simulation.getDeltaT())
                            for _ in range(yellow_duration):
                                traci.simulationStep()
                                step += 1
                                for tid in TRAFFIC_LIGHT_IDS: phase_timers[tid] += traci.simulation.getDeltaT()
                        
                        traci.trafficlight.setPhase(tls_id, target_green_phase)
                    
                    phase_timers[tls_id] = 0.0 

            if total_steps > TRAINING_START_BUFFER_SIZE and total_steps % TRAINING_INTERVAL == 0:
                for tls_id in TRAFFIC_LIGHT_IDS:
                    if len(replay_buffers[tls_id]) >= BATCH_SIZE:
                        batch = random.sample(replay_buffers[tls_id], BATCH_SIZE)
                        agents[tls_id].train_on_batch(batch)
            
            if total_steps > TRAINING_START_BUFFER_SIZE and total_steps % TARGET_UPDATE_FREQ == 0:
                for agent in agents.values():
                    agent.update_target_network()

            for tid in TRAFFIC_LIGHT_IDS: phase_timers[tid] += traci.simulation.getDeltaT()
            step += 1
            total_steps += 1

        traci.close()
        
        if g_epsilon > EPSILON_END:
            g_epsilon *= EPSILON_DECAY

        avg_reward = sum(episode_rewards.values()) / len(TRAFFIC_LIGHT_IDS) if TRAFFIC_LIGHT_IDS else 0
        print(f"Episode {episode+1}/{EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {g_epsilon:.3f} | Total Steps: {total_steps}")

        if (episode + 1) % 10 == 0:
            agents[TRAFFIC_LIGHT_IDS[0]].save_model(LOCAL_MODEL_FILE)
            print(f"Checkpoint saved to {LOCAL_MODEL_FILE}")

    print("--- Local Training Complete ---")