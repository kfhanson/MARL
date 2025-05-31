import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd # For reading CSV
import traceback

# --- DQN Hyperparameters (Keep consistent or tune for offline) ---
STATE_FEATURES = 4
NUM_ACTIONS = 2
SEQUENCE_LENGTH = 1 # If state from CSV is just one snapshot

LEARNING_RATE = 0.0005 # May need smaller LR for offline
GAMMA = 0.95
# Epsilon is not used during offline training in the same way (no exploration)
# but can be used if you were to fine-tune online later.

# For offline training, we typically iterate over the dataset multiple times (epochs)
NUM_EPOCHS = 50
BATCH_SIZE = 64
TARGET_NETWORK_UPDATE_FREQ_OFFLINE = 500 # Batches between target updates

MODEL_SAVE_PATH = "offline_dqn_lstm_agent.weights.h5"
DATA_CSV_FILE = "sumo_offline_rl_data.csv" # From Phase 1

# --- DQNAgent Class (Mostly same as before, minor tweaks for offline context) ---
class DQNAgentOffline:
    def __init__(self, state_dims, action_size, sequence_length):
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE

        self.q_network = self._build_lstm_model()
        self.target_network = self._build_lstm_model() # clone_model(self.q_network) also works after first build
        self.target_network.set_weights(self.q_network.get_weights()) # Initialize

        print("Offline DQN Agent Initialized with LSTM model.")
        self.q_network.summary()

    def _build_lstm_model(self):
        # Same model architecture as before
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
        """
        Trains the Q-network on a provided batch of experiences.
        batch_experiences: list of (state, action, reward, next_state, done) tuples.
        """
        if not batch_experiences:
            return

        # Prepare batch data for network prediction
        current_states_batch_list = []
        next_states_batch_list = []
        for exp in batch_experiences:
            # State from CSV might be (N,S,E,W). Reshape for LSTM.
            state_np = np.array(exp[0], dtype=np.float32).reshape(1, self.sequence_length, self.state_feature_size)
            next_state_np = np.array(exp[3], dtype=np.float32).reshape(1, self.sequence_length, self.state_feature_size)
            current_states_batch_list.append(state_np[0]) # Remove the first batch dim
            next_states_batch_list.append(next_state_np[0]) # Remove the first batch dim
        
        current_states_batch = np.array(current_states_batch_list) # Shape: (BATCH_SIZE, seq_len, features)
        next_states_batch = np.array(next_states_batch_list)     # Shape: (BATCH_SIZE, seq_len, features)


        q_current_state_batch = self.q_network.predict(current_states_batch, verbose=0)
        q_next_state_target_batch = self.target_network.predict(next_states_batch, verbose=0)

        for i, (state_tuple, action, reward, next_state_tuple, done) in enumerate(batch_experiences):
            target_q_value = reward
            if not done:
                target_q_value += self.gamma * np.amax(q_next_state_target_batch[i])
            
            q_current_state_batch[i][int(action)] = target_q_value # Ensure action is int
        
        history = self.q_network.fit(current_states_batch, q_current_state_batch, batch_size=len(batch_experiences), epochs=1, verbose=0)
        return history.history['loss'][0]


    def load_model(self, filepath): # Same
        try:
            self.q_network.load_weights(filepath)
            self.update_target_network()
            print(f"Model weights loaded from {filepath}")
        except Exception as e: print(f"Error loading model weights: {e}")

    def save_model(self, filepath): # Same
        try:
            self.q_network.save_weights(filepath)
            print(f"Model weights saved to {filepath}")
        except Exception as e: print(f"Error saving model weights: {e}")


def load_offline_data(csv_filepath):
    """Loads experiences from the CSV file."""
    df = pd.read_csv(csv_filepath)
    experiences = []
    for _, row in df.iterrows():
        state = (row['state_N'], row['state_S'], row['state_E'], row['state_W'])
        action = int(row['action'])
        reward = float(row['reward'])
        next_state = (row['next_state_N'], row['next_state_S'], row['next_state_E'], row['next_state_W'])
        done = bool(row['done'])
        experiences.append((state, action, reward, next_state, done))
    print(f"Loaded {len(experiences)} experiences from {csv_filepath}")
    return experiences


def train_offline_agent(total_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH):
    agent = DQNAgentOffline(STATE_FEATURES, NUM_ACTIONS, SEQUENCE_LENGTH)
    # agent.load_model(model_save_path) # For continuing training

    experiences_dataset = load_offline_data(DATA_CSV_FILE)
    if not experiences_dataset:
        print("No data to train on. Exiting.")
        return

    total_batches_trained = 0
    for epoch in range(total_epochs):
        random.shuffle(experiences_dataset) # Shuffle data each epoch
        epoch_loss = 0
        num_batches_in_epoch = 0

        for i in range(0, len(experiences_dataset), BATCH_SIZE):
            batch = experiences_dataset[i:i + BATCH_SIZE]
            if len(batch) < BATCH_SIZE // 2 and epoch < total_epochs -1 : # Skip tiny trailing batches unless last epoch
                continue
            
            loss = agent.train_on_batch(batch)
            epoch_loss += loss
            num_batches_in_epoch +=1
            total_batches_trained += 1

            if total_batches_trained % TARGET_NETWORK_UPDATE_FREQ_OFFLINE == 0:
                agent.update_target_network()
                print(f"  Target network updated at batch {total_batches_trained}")
        
        avg_epoch_loss = epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        print(f"Epoch {epoch + 1}/{total_epochs} completed. Avg Loss: {avg_epoch_loss:.6f}")

        if (epoch + 1) % 10 == 0: # Save periodically
            agent.save_model(model_save_path)
    
    agent.save_model(model_save_path) # Final save
    print("Offline training complete. Model saved.")


if __name__ == "__main__":
    # --- Run Phase 1: Data Collection (if needed) ---
    # print("Starting Phase 1: Data Collection...")
    # Make sure to uncomment the run_sumo_and_log_data call in sumo_data_logger.py
    # and run that script first if you don't have the CSV.
    # For this script, we assume sumo_offline_rl_data.csv already exists.

    # --- Run Phase 2: Offline Training ---
    print("\nStarting Phase 2: Offline DQN Training...")
    train_offline_agent(total_epochs=NUM_EPOCHS) # Adjust epochs as needed