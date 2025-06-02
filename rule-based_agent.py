import os
import sys
import traci
import random
import traceback

try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary, net
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")

SUMO_BINARY = checkBinary('sumo') 
CONFIG_FILE = "osm.sumocfg" 
NET_FILE = "osm.net.xml"
TRAFFIC_LIGHT_ID = 'joinedS_6421159832_cluster_3639980474_3640024452_3640024453_6421159827_#4more_cluster_6421159831_7129012339'

CONGESTION_THRESHOLD_POLICY = 5
MIN_GREEN_TIME_IOT_POLICY = 10
YELLOW_TIME_IOT_POLICY = 6 

AGENT_ACTION_TO_SUMO_GREEN_PHASE = {0: 0, 1: 2, 2: 4, 3: 6} # S, E, N, W
SUMO_GREEN_PHASE_TO_AGENT_ACTION = {v: k for k, v in AGENT_ACTION_TO_SUMO_GREEN_PHASE.items()}
SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7}

# Edges for policy decision making
IOT_POLICY_EDGE_TO_DIRECTION_NAME = {
    "754598165#2": "north",
    "1053267667#3": "south",
    "749662140#0": "east",
    "885403818#2": "west",
}
# Map policy-understood directions to agent actions
DIRECTION_TO_AGENT_ACTION_POLICY = {
    "south": 0, "east":  1, "north": 2, "west":  3,
}

g_net_obj_policy_rb = None 

def get_phase_green_directions_for_policy_rb(tls_id, phase_state_str):
    global g_net_obj_policy_rb
    if g_net_obj_policy_rb is None: return set()
    green_directions = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        if not controlled_links: return green_directions
        for i, state_char in enumerate(phase_state_str):
            if state_char.lower() == 'g':
                if i < len(controlled_links) and controlled_links[i]:
                    first_connection_in_link = controlled_links[i][0]
                    if first_connection_in_link:
                        from_lane_id = first_connection_in_link[0]
                        from_lane_obj = g_net_obj_policy_rb.getLane(from_lane_id)
                        if from_lane_obj:
                            from_edge_id = from_lane_obj.getEdge().getID()
                            if from_edge_id in IOT_POLICY_EDGE_TO_DIRECTION_NAME:
                                green_directions.add(IOT_POLICY_EDGE_TO_DIRECTION_NAME[from_edge_id])
    except KeyError as e_key: print(f"Warning: Lane ID {e_key} not found.")
    except Exception as e_phase_dir: print(f"Warning: Error in get_phase_green_directions_for_policy_rb: {e_phase_dir}")
    return green_directions

def get_rule_based_policy_action_rb(current_tl_sumo_phase_idx):
    try:
        policy_approach_congestion = {}
        for edge_id, direction_name in IOT_POLICY_EDGE_TO_DIRECTION_NAME.items():
            count = traci.edge.getLastStepHaltingNumber(edge_id)
            policy_approach_congestion[direction_name] = count

        program_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT_ID)[0]
        current_phase_def = program_logic.phases[current_tl_sumo_phase_idx]
        current_phase_state_str = current_phase_def.state
        
        directions_served_by_current_green = get_phase_green_directions_for_policy_rb(TRAFFIC_LIGHT_ID, current_phase_state_str)
        current_agent_action = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx)
        
        sorted_congestion = sorted(policy_approach_congestion.items(), key=lambda item: item[1], reverse=True)

        if 'g' in current_phase_state_str.lower():
            is_current_serving_high_congestion = False
            for d in directions_served_by_current_green:
                if policy_approach_congestion.get(d, 0) >= CONGESTION_THRESHOLD_POLICY:
                    is_current_serving_high_congestion = True; break
            if is_current_serving_high_congestion:
                # If multiple directions served, check if any of them is the top congested
                is_serving_top_congested = False
                if directions_served_by_current_green and sorted_congestion:
                    top_congested_dir_name = sorted_congestion[0][0]
                    if top_congested_dir_name in directions_served_by_current_green:
                        is_serving_top_congested = True
                
                if is_serving_top_congested: # If serving the most congested, stay
                    return current_agent_action

            # If not serving the most congested, or if current congestion is low, look for a switch.
            for direction_name, count in sorted_congestion:
                if count >= CONGESTION_THRESHOLD_POLICY and direction_name not in directions_served_by_current_green:
                    return DIRECTION_TO_AGENT_ACTION_POLICY.get(direction_name)
            # If no other direction is significantly congested, or current is okay, stay
            return current_agent_action 
        else:
            # Find the most congested direction and switch to it
            for direction_name, count in sorted_congestion:
                if count >= CONGESTION_THRESHOLD_POLICY:
                    return DIRECTION_TO_AGENT_ACTION_POLICY.get(direction_name)
            # If no significant congestion, default to serving the one with highest count (even if low) or a default
            return DIRECTION_TO_AGENT_ACTION_POLICY.get(sorted_congestion[0][0]) if sorted_congestion else 0
    except Exception as e_policy_action:
        print(f"Error in get_rule_based_policy_action_rb: {e_policy_action}")
        return random.choice(list(AGENT_ACTION_TO_SUMO_GREEN_PHASE.keys()))


def execute_rule_based_simulation(simulation_duration_steps=3600, seed=43):
    global g_net_obj_policy_rb
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE]
    sumo_cmd_list.extend([
        "--tripinfo-output", "tripinfo_rule_based.xml",
        "--summary-output", "summary_rule_based.xml",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--no-step-log", "true",
        "--seed", str(seed)
    ])
    if SUMO_BINARY.endswith("-gui"): sumo_cmd_list.append("--step-length=0.2")

    traci.start(sumo_cmd_list)
    g_net_obj_policy_rb = net.readNet(NET_FILE)
    print("SUMO started for Rule-Based Agent baseline run.")

    current_step = 0
    phase_decision_timer = 0.0

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and current_step < simulation_duration_steps:
            current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            is_currently_green = current_tl_sumo_phase_idx in AGENT_ACTION_TO_SUMO_GREEN_PHASE.values()

            if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME_IOT_POLICY:
                decided_agent_action = get_rule_based_policy_action_rb(current_tl_sumo_phase_idx)

                if decided_agent_action is not None:
                    target_sumo_green_phase = AGENT_ACTION_TO_SUMO_GREEN_PHASE[decided_agent_action]
                    if current_tl_sumo_phase_idx != target_sumo_green_phase:
                        if current_tl_sumo_phase_idx in SUMO_GREEN_TO_YELLOW_PHASE:
                            yellow_phase_idx = SUMO_GREEN_TO_YELLOW_PHASE[current_tl_sumo_phase_idx]
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                            for _ in range(int(YELLOW_TIME_IOT_POLICY / traci.simulation.getDeltaT())):
                                if traci.simulation.getMinExpectedNumber() == 0: break
                                traci.simulationStep(); current_step += 1
                            if traci.simulation.getMinExpectedNumber() == 0: break
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                    phase_decision_timer = 0.0
            
            traci.simulationStep()
            current_step += 1
            phase_decision_timer += traci.simulation.getDeltaT()

            if traci.simulation.getMinExpectedNumber() == 0: break
        
        print(f"Rule-Based Agent simulation finished. Steps: {current_step}.")

    except traci.exceptions.FatalTraCIError as e: print(f"Fatal TraCI Error: {e}")
    except KeyboardInterrupt: print("Simulation interrupted.")
    except Exception as e:
        print(f"Unexpected Python error:")
        traceback.print_exc()
    finally:
        if 'traci' in sys.modules and traci.isLoaded() and traci.getConnection():
            try:
                traci.close()
                print("TraCI connection closed.")
            except traci.exceptions.TraCIException as e_traci_close:
                print(f"TraCI warning on close: {e_traci_close}")
        else:
            print("No active TraCI connection to close or traci module not fully loaded for check.")


if __name__ == "__main__":
    execute_rule_based_simulation(simulation_duration_steps=3600, seed=43)
    print("Rule-based baseline files (tripinfo_rule_based.xml, summary_rule_based.xml) should be generated.")