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