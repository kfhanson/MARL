import itertools
import os
import json
import time
from sumolib import checkBinary

from refactored_trainer import train_offline_agent_with_params
from refactored_evaluator import evaluate_model_in_sumo 

# --- Grid Search Configuration ---
DATA_CSV = "sumo_data.csv"
MODEL_SAVE_DIRECTORY = "grid_search_models"
os.makedirs(MODEL_SAVE_DIRECTORY, exist_ok=True)

# Define the grid of hyperparameters to search
param_grid = {
    'learning_rate': [0.001, 0.0005], 
    'gamma': [0.95],                  
    'num_epochs': [75, 100], 
    'batch_size': [64, 128],
}

SUMO_BINARY_PATH = checkBinary('sumo')
CONFIG_FILE_PATH = "osm.sumocfg"
NET_FILE_PATH = "osm.net.xml"
TRAFFIC_LIGHT_ID_EVAL = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'
ACTION_TO_GREEN_PHASE_EVAL = {0: 0, 1: 2, 2: 4, 3: 6}
GREEN_TO_YELLOW_PHASE_EVAL = {0: 1, 2: 3, 4: 5, 6: 7}
MIN_GREEN_TIME_EVAL = 10
YELLOW_TIME_EVAL = 6
APPROACH_EDGES_MAP_EVAL = {
    "north": "754598165#2", "south": "1053267667#3",
    "east": "749662140#0", "west": "885403818#2",
}
VEHICLE_BINS_LIST_EVAL = [5, 15, 30]
STATE_FEATURES_EVAL_GRID = 4
NUM_ACTIONS_EVAL_GRID = 4
SEQUENCE_LENGTH_EVAL_GRID = 1

def perform_grid_search():
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    best_metric = float('inf')
    best_config = None

    print(f"Starting Grid Search with {len(experiments)} experiments...\n")

    for i, config in enumerate(experiments):
        start_time = time.time()
        print(f"--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Config: {config}")

        model_filename = f"model_lr{config['learning_rate']}_gamma{config.get('gamma',0.95)}_ep{config['num_epochs']}_bs{config['batch_size']}_trial{i}.weights.h5"
        model_save_path = os.path.join(MODEL_SAVE_DIRECTORY, model_filename)

        # 1. Train the model with current config
        training_success = train_offline_agent_with_params(
            data_csv_file=DATA_CSV,
            model_save_path=model_save_path,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            state_features=STATE_FEATURES_EVAL_GRID, 
            num_actions=NUM_ACTIONS_EVAL_GRID,
            sequence_length=SEQUENCE_LENGTH_EVAL_GRID,
            target_update_freq=500
        )

        if not training_success:
            print("Training failed for this configuration. Skipping evaluation.")
            eval_metric = float('inf')
        else:
            # 2. Evaluate the trained model
            eval_metric = evaluate_model_in_sumo(
                model_load_path=model_save_path,
                sumo_binary_path=SUMO_BINARY_PATH,
                config_file_path=CONFIG_FILE_PATH,
                net_file_path=NET_FILE_PATH,
                traffic_light_id=TRAFFIC_LIGHT_ID_EVAL,
                action_to_green_phase_map=ACTION_TO_GREEN_PHASE_EVAL,
                green_to_yellow_phase_map=GREEN_TO_YELLOW_PHASE_EVAL,
                min_green_time_val=MIN_GREEN_TIME_EVAL,
                yellow_time_val=YELLOW_TIME_EVAL,
                approach_edges_map=APPROACH_EDGES_MAP_EVAL,
                vehicle_bins_list=VEHICLE_BINS_LIST_EVAL,
                state_features_val=STATE_FEATURES_EVAL_GRID,
                num_actions_val=NUM_ACTIONS_EVAL_GRID,
                sequence_length_val=SEQUENCE_LENGTH_EVAL_GRID,
                num_eval_episodes=3
            )
        
        print(f"Evaluation Metric (e.g., Avg Wait Time): {eval_metric:.4f}")
        
        results.append({
            'config': config,
            'eval_metric': eval_metric,
            'model_path': model_save_path
        })

        if eval_metric < best_metric:
            best_metric = eval_metric
            best_config = config
            print(f"*** New Best Metric Found: {best_metric:.4f} with config: {best_config} ***")
        
        end_time = time.time()
        print(f"Experiment {i+1} took {end_time - start_time:.2f} seconds.\n")

    # --- Output Results ---
    print("\n--- Grid Search Complete ---")
    results.sort(key=lambda x: x['eval_metric']) 

    print("\nTop Results:")
    for res in results[:5]: # Print top 5
        print(f"Metric: {res['eval_metric']:.4f}, Config: {res['config']}, Model: {res['model_path']}")

    if best_config: 
        print("\nBest Overall Configuration:")
        best_result_entry = results[0]
        print(f"Metric: {best_result_entry['eval_metric']:.4f}")
        print(f"Config: {best_result_entry['config']}")
        print(f"Best model saved at: {best_result_entry['model_path']}")
    else:
        print("No successful configurations found.")

    # Save all results
    with open("grid_search_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nFull grid search results saved to grid_search_results.json")

if __name__ == "__main__":
    perform_grid_search()