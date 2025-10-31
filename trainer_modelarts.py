# Will be deployed to ModelArts for training

import os
import sys
import random
import numpy as np
import argparse
import pandas as pd
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam #

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

        self.q_network.summary()
    
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
        if not batch_experiences:
            return 0.0
        
        current_states_list = [exp[0] for exp in batch_experiences]
        next_states_list = [exp[3] for exp in batch_experiences]

        current_states_batch = np.array(current_states_list).reshape(-1, self.sequence_length, self.state_feature_size)
        next_states_batch = np.array(next_states_list).reshape(-1, self.sequence_length, self.state_feature_size)

        q_current_state_batch = self.q_network.predict(current_states_batch, verbose=0)
        q_next_state_target_batch = self.target_network.predict(next_states_batch, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(batch_experiences):
            target_q_value = reward
            if not done:
                target_q_value += self.gamma * np.amax(q_next_state_target_batch[i])
            q_current_state_batch[i][action] = target_q_value
        
        history = self.q_network.fit(current_states_batch, q_current_state_batch, batch_size=len(batch_experiences), epochs=1, verbose=0)
        return history.history['loss'][0]
    
    def load_model(self, filepath):
        self.q_network.load_weights(filepath)
        self.update_target_network()
        print(f"Model loaded from {filepath}")
    
    def save_model(self, filepath):
        self.q_network.save_weights(filepath)
        print(f"Model saved to {filepath}")

def load_training_data(data_dir, state_features):
    experiences = []
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return experiences
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)

            state_cols = [f"state_{i}" for i in range(state_features)]
            next_state_cols = [f"next_state_{i}" for i in range(state_features)]

            for _, row in df.iterrows():
                try: 
                    state = row[state_cols].values.astype(np.float32).tolist()
                    action = int(row['action'])
                    reward = float(row['reward'])
                    next_state = row[next_state_cols].values.astype(np.float32).tolist()
                    done = bool(row['done'])

                    experiences.append((state, action, reward, next_state, done))
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid row in {filepath}: {e}")
                    continue

    print(f"Loaded {len(experiences)} experiences from {data_dir}")
    return experiences

if __name__ == "__main__":
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    # 2. ModelArts I/O paths
    data_input_dir = os.environ.get("DLS_DATA_URL", "/")
    model_output_dir = os.environ.get("DLS_TRAIN_URL", "/")

    # 3. Define agent
    agent = DQNAgent(
        state_dims = #,
        action_size= #,
        sequence_length=1,
        learning_rate=#,
        gamma=#
    )

    # 4. Load previous model if exists
    model_filename = "flow_model_latest.weights.h5"
    model_load_path = os.path.join(data_input_dir, model_filename)

    if os.path.exists(model_load_path):
        try:
            agent.load_model(model_load_path)
        except Exception as e:
            print(f"Failed to load model from {model_load_path}: {e}")
    else:
        print(f"No existing model found at {model_load_path}. Starting fresh.")

    # 5. Load experience data
    experience_dataset = load_training_data(data_input_dir, )