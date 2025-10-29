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

EDGE_MAP = {
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
VEHICLE_BINS = [5, 15, 30]
