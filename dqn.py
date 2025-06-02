import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd
import traceback

# --- DQN Hyperparameters ---
STATE_FEATURES = 4  # N, S, E, W discretized congestion counts
NUM_ACTIONS = 4     # Now 4 actions: Serve S, E, N, W (maps to SUMO phases 0, 2, 4, 6)
SEQUENCE_LENGTH = 1 # Using current state as a sequence of length 1 for LSTM

LEARNING_RATE = 0.0005
GAMMA = 0.95
NUM_EPOCHS = 100
BATCH_SIZE = 64
# Frequency to update target network (in terms of number of batches processed)
TARGET_NETWORK_UPDATE_FREQ_OFFLINE = 500

MODEL_SAVE_PATH = "offline_dqn_agent.weights.h5"
DATA_CSV_FILE = "sumo_data.csv"

class DQNAgentOffline:
    def __init__(self, state_dims, action_size, sequence_length):
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length

        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE

        self.q_network = self._build_lstm_model()
        self.target_network = self._build_lstm_model()
        self.update_target_network()

        print(f"Offline DQN Agent Initialized. State Features: {state_dims}, Actions: {action_size}")
        self.q_network.summary()

    def _build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(32, activation='relu', return_sequences=False), 
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def train_on_batch(self, batch_experiences):
        if not batch_experiences:
            return 0.0

        current_states_batch_list = []
        next_states_batch_list = []

        for exp_state, _, _, exp_next_state, _ in batch_experiences:
            state_np = np.array(exp_state, dtype=np.float32).reshape(1, self.sequence_length, self.state_feature_size)
            next_state_np = np.array(exp_next_state, dtype=np.float32).reshape(1, self.sequence_length, self.state_feature_size)
            current_states_batch_list.append(state_np[0]) 
            next_states_batch_list.append(next_state_np[0])
        
        current_states_batch = np.array(current_states_batch_list)
        next_states_batch = np.array(next_states_batch_list)

        q_current_state_batch = self.q_network.predict(current_states_batch, verbose=0)
        q_next_state_target_batch = self.target_network.predict(next_states_batch, verbose=0)

        for i, (state_tuple, action, reward, next_state_tuple, done) in enumerate(batch_experiences):
            target_q_value = reward
            if not done:
                target_q_value += self.gamma * np.amax(q_next_state_target_batch[i])
            
            action_idx = int(action)
            if 0 <= action_idx < self.action_size:
                 q_current_state_batch[i][action_idx] = target_q_value
            else:
                print(f"Warning: Invalid action {action} encountered in batch. Skipping update for this sample.")


        history = self.q_network.fit(current_states_batch, q_current_state_batch, batch_size=len(batch_experiences), epochs=1, verbose=0)
        return history.history['loss'][0]

    def load_model(self, filepath):
        try:
            self.q_network.load_weights(filepath)
            self.update_target_network()
            print(f"Model weights loaded from {filepath}")
        except Exception as e: print(f"Error loading model weights: {e}")

    def save_model(self, filepath):
        try:
            self.q_network.save_weights(filepath)
            print(f"Model weights saved to {filepath}")
        except Exception as e: print(f"Error saving model weights: {e}")


def load_offline_data_from_csv(csv_filepath):
    """Loads experiences (s, a, r, s', done) from the CSV file."""
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Data CSV file not found at {csv_filepath}")
        return []
        
    experiences = []
    required_cols = ["state_N", "state_S", "state_E", "state_W", "action", "reward",
                     "next_state_N", "next_state_S", "next_state_E", "next_state_W", "done"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing one or more required columns: {required_cols}")
        return []

    for _, row in df.iterrows():
        try:
            state = (
                int(row['state_N']), int(row['state_S']),
                int(row['state_E']), int(row['state_W'])
            )
            action = int(row['action'])
            if not (0 <= action < NUM_ACTIONS):
                continue

            reward = float(row['reward'])
            next_state = (
                int(row['next_state_N']), int(row['next_state_S']),
                int(row['next_state_E']), int(row['next_state_W'])
            )
            done_val = row['done']
            if isinstance(done_val, str):
                done = done_val.lower() == 'true'
            else:
                done = bool(done_val)
                
            experiences.append((state, action, reward, next_state, done))
        except ValueError as e:
            print(f"Warning: Skipping row due to data conversion error: {e}. Row: {row.to_dict()}")
            continue
            
    print(f"Loaded {len(experiences)} valid experiences from {csv_filepath}")
    return experiences


def train_offline_agent(total_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH):
    agent = DQNAgentOffline(STATE_FEATURES, NUM_ACTIONS, SEQUENCE_LENGTH)
    # agent.load_model(model_save_path) # Continue training a saved model

    experiences_dataset = load_offline_data_from_csv(DATA_CSV_FILE)
    if not experiences_dataset:
        print("No data to train on or error loading data. Exiting.")
        return

    total_batches_trained_ever = 0
    for epoch in range(total_epochs):
        random.shuffle(experiences_dataset)
        epoch_total_loss = 0.0
        num_batches_this_epoch = 0

        for i in range(0, len(experiences_dataset), BATCH_SIZE):
            batch = experiences_dataset[i:i + BATCH_SIZE]
            # Only train if batch is reasonably full, especially for early epochs
            if len(batch) < BATCH_SIZE and epoch < total_epochs - 5 : # Allow smaller last batch for last few epochs
                 if len(batch) < BATCH_SIZE // 2 : continue # skip very small batches early on

            loss = agent.train_on_batch(batch)
            epoch_total_loss += loss
            num_batches_this_epoch +=1
            total_batches_trained_ever += 1

            if total_batches_trained_ever % TARGET_NETWORK_UPDATE_FREQ_OFFLINE == 0:
                agent.update_target_network()
                print(f"  Target network updated at total batch {total_batches_trained_ever}")
        
        avg_epoch_loss = epoch_total_loss / num_batches_this_epoch if num_batches_this_epoch > 0 else 0
        print(f"Epoch {epoch + 1}/{total_epochs} completed. Avg Batch Loss: {avg_epoch_loss:.6f}")

        if (epoch + 1) % 10 == 0 or epoch == total_epochs -1 :
            agent.save_model(model_save_path)
            print(f"Model saved after epoch {epoch+1}")
    
    print("Offline training complete. Final model saved.")

if __name__ == "__main__":
    print("Starting Offline DQN Training for 4-Action Traffic Light...")
    train_offline_agent(total_epochs=NUM_EPOCHS)