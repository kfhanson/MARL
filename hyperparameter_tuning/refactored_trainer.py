import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os

# --- Default Hyperparameters (can be overridden by grid search) ---
DEFAULT_STATE_FEATURES = 4
DEFAULT_NUM_ACTIONS = 4
DEFAULT_SEQUENCE_LENGTH = 1
DEFAULT_GAMMA = 0.95
DEFAULT_TARGET_UPDATE_FREQ = 500 # Batches

# DATA_CSV_FILE will be passed as an argument or configured globally
# MODEL_SAVE_DIR will be passed to save models from different trials

class DQNAgentOffline:
    def __init__(self, state_dims, action_size, sequence_length, learning_rate, gamma): 
        self.state_feature_size = state_dims
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = gamma 
        self.learning_rate = learning_rate 

        self.q_network = self._build_lstm_model()
        self.target_network = self._build_lstm_model()
        self.update_target_network()

    def _build_lstm_model(self):
        lstm_units = 32 
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(lstm_units, activation='relu', return_sequences=False),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def train_on_batch(self, batch_experiences):
        if not batch_experiences: return 0.0
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
        history = self.q_network.fit(current_states_batch, q_current_state_batch, batch_size=len(batch_experiences), epochs=1, verbose=0)
        return history.history['loss'][0]

    def save_model(self, filepath):
        try:
            self.q_network.save_weights(filepath)
            # print(f"Model weights saved to {filepath}")
        except Exception as e: print(f"Error saving model weights: {e}")
    
def load_offline_data_from_csv(csv_filepath, num_actions_expected):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError: return []
    experiences = []
    required_cols = ["state_N", "state_S", "state_E", "state_W", "action", "reward",
                     "next_state_N", "next_state_S", "next_state_E", "next_state_W", "done"]
    if not all(col in df.columns for col in required_cols): return []
    for _, row in df.iterrows():
        try:
            state = (int(row['state_N']), int(row['state_S']), int(row['state_E']), int(row['state_W']))
            action = int(row['action'])
            if not (0 <= action < num_actions_expected): continue
            reward = float(row['reward'])
            next_state = (int(row['next_state_N']), int(row['next_state_S']), int(row['next_state_E']), int(row['next_state_W']))
            done_val = row['done']
            if isinstance(done_val, str): done = done_val.lower() == 'true'
            else: done = bool(done_val)
            experiences.append((state, action, reward, next_state, done))
        except ValueError: continue
    return experiences

# --- Parameterized Training Function ---
def train_offline_agent_with_params(
    data_csv_file, model_save_path,
    state_features=DEFAULT_STATE_FEATURES, num_actions=DEFAULT_NUM_ACTIONS, sequence_length=DEFAULT_SEQUENCE_LENGTH,
    learning_rate=0.001, gamma=DEFAULT_GAMMA, num_epochs=50, batch_size=64,
    target_update_freq=DEFAULT_TARGET_UPDATE_FREQ
):
    print(f"Training with LR={learning_rate}, Gamma={gamma}, Epochs={num_epochs}, BS={batch_size}")
    agent = DQNAgentOffline(state_features, num_actions, sequence_length, learning_rate, gamma)
    
    experiences_dataset = load_offline_data_from_csv(data_csv_file, num_actions)
    if not experiences_dataset:
        print(f"No data loaded from {data_csv_file}. Aborting training for this config.")
        return False 

    total_batches_trained_ever = 0
    for epoch in range(num_epochs):
        random.shuffle(experiences_dataset)
        for i in range(0, len(experiences_dataset), batch_size):
            batch = experiences_dataset[i:i + batch_size]
            if len(batch) < batch_size // 2 and epoch < num_epochs -1 : continue
            
            loss = agent.train_on_batch(batch)
            total_batches_trained_ever += 1

            if total_batches_trained_ever % target_update_freq == 0:
                agent.update_target_network()

    agent.save_model(model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    return True