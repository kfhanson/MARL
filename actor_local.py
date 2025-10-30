import os
import sys
import traci
import numpy as np
import random
import tensorflow as tf 
import keras
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam #
import traceback

try:
    from obs import ObsClient
except ImportError:
    print("Huawei OBS SDK not found.")
    sys.exit(1)

OBS_ACCESS_KEY = ""
OBS_SECRET_KEY = ""
OBS_ENDPOINT = ""
MODEL_BUCKET_NAME = ""
DATA_BUCKET_NAME = ""
MODEL_FILENAME = ""

SUMO_BINARY = "sumo"
CONFIG_FILE = "bsd.sumocfg"
TRAFFIC_LIGHT_IDS = ["cluster_12705056632_3639980474_3640024452_3640024453_#7more", "cluster_3640024470_3640024471_3640024476_699593339_#8more"]

NEIGHBOR_EDGE_MAP = {
    "cluster_12705056632_3639980474_3640024452_3640024453_#7more": {
        "neighbor_incoming_edge"
    }
}

# Agent/Environment Parameters
STATE_FEATURES = 5
NUM_ACTIONS = 4
SEQUENCE_LENGTH = 1

# Data Collection Parameters
EPISODES_TO_RUN = 5
MAX_STEPS_PER_EPISODE = 3600
EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
g_epsilon = EPSILON_START

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

def get_obs_client():
    return ObsClient(
        access_key_id=OBS_ACCESS_KEY,
        secret_access_key=OBS_SECRET_KEY,
        server=OBS_ENDPOINT
    )

def download_model_from_obs(local_path):
    obs_client = get_obs_client()
    try:
        obs_client = get_obs_client()
        resp = obs_client.getObject(MODEL_BUCKET_NAME, MODEL_FILENAME, downloadPath=local_path)
        if resp.status < 300:
            print("Model downloaded successfully to", local_path)
            return True
        else:
            print("Failed to download model. Status:", resp.status)
            return False
    except Exception as e:
        print("Error downloading model from OBS:", e)
        return False

def upload_data_to_obs(local_csv_path):
    object_key = f"experiences_{int(time.time())}.csv"
    try:
        obs_client = get_obs_client()
        resp = obs_client.putFile(DATA_BUCKET_NAME, object_key, file_path=local_csv_path)
        if resp.status < 300:
            print("Data uploaded successfully to OBS as", object_key)
        else:
            print("Failed to upload data. Status:", resp.status)
    except Exception as e:
        print("Error uploading data to OBS:", e)


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
    

class ActorAgent:
    def __init__(self, state_dims, action_size, sequence_length):
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.q_network = self._build_lstm_model()
    
    def _build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def select_action(self, current_state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        if current_state is None:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(current_state, verbose=0)
        return np.argmax(q_values[0])