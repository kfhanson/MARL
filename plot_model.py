import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


STATE_FEATURES = 4  # N, S, E, W discretized congestion counts
NUM_ACTIONS = 4     # Serve S (0), E (1), N (2), W (3)
SEQUENCE_LENGTH = 1 # Using current state as a sequence of length 1 for LSTM
LSTM_UNITS = 32     # Number of units in the LSTM layer (must match trained model)

MODEL_WEIGHTS_PATH = "grid_search_models\model_lr0.0005_gamma0.95_ep100_bs128_trial7.weights.h5"
OUTPUT_PLOT_FILENAME = "dqn_lstm_model_architecture.png"

def build_model_for_plotting(state_features, num_actions, sequence_length, lstm_units):
    """
    Rebuilds the Keras model architecture.
    This MUST EXACTLY MATCH the architecture used during training.
    """
    model = Sequential([
        Input(shape=(sequence_length, state_features), name="Input_State_Sequence"),
        LSTM(lstm_units, activation='relu', name="LSTM_Layer"),
        Dense(num_actions, activation='linear', name="Output_Q_Values")
    ])
    return model

def plot_nn_model():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights file not found at '{MODEL_WEIGHTS_PATH}'.")
        print("Please ensure the path is correct and the model has been trained and saved.")
        return

    print(f"Attempting to plot model architecture. Weights will be loaded from: {MODEL_WEIGHTS_PATH}")
    model = build_model_for_plotting(STATE_FEATURES, NUM_ACTIONS, SEQUENCE_LENGTH, LSTM_UNITS)
    print("\nModel architecture successfully rebuilt (before loading weights).")
    model.summary()
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print(f"\nSuccessfully loaded weights from {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"\nWarning: Could not load weights from {MODEL_WEIGHTS_PATH}. Plotting structure only. Error: {e}")
        print("This is usually fine for just plotting the architecture diagram.")

    # Plot the model
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=OUTPUT_PLOT_FILENAME,
            show_shapes=True,         # Shows input/output shapes of layers
            show_dtype=False,         # Optionally show data types
            show_layer_names=True,    # Display layer names
            rankdir="TB",             # TB: Top-to-Bottom, LR: Left-to-Right
            expand_nested=False,      # Whether to expand nested models
            dpi=96,                   # Resolution of the output image
            show_layer_activations=True # Shows activation functions in layer blocks
        )
        print(f"\nModel architecture plot saved to {OUTPUT_PLOT_FILENAME}")
        print("Please ensure Graphviz and pydot are installed and configured in your system's PATH if you see errors related to them.")
    except ImportError as e_imp:
         print(f"\nError generating plot: {e_imp}.")
         print("Make sure you have 'pydot' and 'graphviz' installed.")
         print("Try: pip install pydot graphviz")
         print("You might also need to install Graphviz at the system level (e.g., 'sudo apt-get install graphviz' or download from graphviz.org).")
    except Exception as e:
        print(f"\nAn unexpected error occurred while plotting the model: {e}")
        print("Ensure Graphviz executables (like 'dot') are in your system's    PATH.")

if __name__ == "__main__":
    plot_nn_model()