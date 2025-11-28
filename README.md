# Intelligent Traffic Corridor: A Multi-Agent Reinforcement Learning Approach

![alt text](https://img.shields.io/badge/python-3.9+-blue.svg)
![alt text](https://img.shields.io/badge/framework-TensorFlow_2.x-orange.svg)
![alt text](https://img.shields.io/badge/simulator-SUMO-green.svg)
![alt text](https://img.shields.io/badge/cloud-Huawei_Cloud-red.svg)

This project demonstrates a prototype for an intelligent, adaptive traffic signal control system for an entire urban corridor. It moves beyond single-intersection optimization by implementing a network of cooperative Multi-Agent Reinforcement Learning (MARL) agents. The system is built on a scalable, cloud-native architecture, leveraging Huawei Cloud ModelArts for distributed training and SUMO for realistic traffic simulation.

### Project Vision & Problem Statement
Traditional traffic light systems rely on fixed timers or simple rule-based logic. This leads to inefficient traffic flow, increased congestion, and unnecessary vehicle idling, especially in corridors where the traffic at one intersection directly impacts the next.
This project tackles this problem by treating each traffic light as an intelligent agent that learns to cooperate with its neighbors. 

The goal is to optimize traffic flow for the entire corridor, not just a single intersection, leading to reduced travel times and vehicle emissions. The core innovation lies in the decoupled "Actor-Learner" architecture, where simulation and data collection can occur on edge devices (simulated locally) while intensive model training is offloaded to a powerful cloud AI platform.

### System Architecture: The "Local Actor, Cloud Learner" Model
This prototype employs a distributed architecture that separates the task of environmental interaction (Acting) from model training (Learning). This is a scalable and efficient paradigm for complex reinforcement learning tasks.

### Component Roles:
- SUMO (Simulation of Urban MObility): Provides a high-fidelity, microscopic simulation of the multi-intersection traffic corridor.
- The Actor (actor_local.py): Runs on a local machine (simulating an edge device). It loads the latest agent model from the cloud, controls the traffic lights in SUMO, and collects valuable (state, action, reward, next_state) experience data.
- Huawei Cloud OBS (Object Storage Service): Acts as the central data lake, storing the experience datasets uploaded by the Actor and the trained model weights produced by the Learner.
- The Huawei ModelArts Learner (trainer_modelarts.py): A powerful training script executed as a Huawei Cloud ModelArts Training Job. It reads the collected data from OBS, trains the Deep Q-Network on high-performance GPUs, and saves the improved model back to OBS, ready for the Actor to download.

## 🧠 Core Technical Concepts (MARL Implementation)

### 1. Multi-Agent Reinforcement Learning (MARL)

Instead of a single agent, we have multiple independent agents (one for each intersection) learning simultaneously in a shared environment. These agents learn a cooperative policy by utilizing a hybrid reward function and a state representation that includes information about their immediate neighbors.

### 2. Deep Q-Network (DQN) with LSTM

Each agent uses a **Deep Q-Network (DQN)** to learn its optimal policy. The key network features are:
*   **LSTM Layer:** The inclusion of an LSTM (Long Short-Term Memory) layer allows the agent to potentially capture temporal dependencies and complex, historical patterns in traffic flow, leading to more predictive actions.
*   **Parameter Sharing:** All agents share the same model weights. This is a crucial technique that stabilizes and accelerates training by allowing all agents' experiences to contribute to a single, robust model.

### 3. Standardized State Representation with E2 Sensors

To eliminate bias caused by varying road lengths or simulation configurations, the system uses SUMO's **E2 Lane Area Detectors**.
*   **Critical Zone:** We define a standardized **30-meter "critical zone"** before each approach.
*   **Agent State:** The agent's state is derived from the **vehicle count** in these zones. This ensures a fair, consistent, and density-focused view of traffic across all intersections in the corridor.

### 4. Hybrid Cooperative Reward Function

To explicitly encourage cooperation and global optimization, an agent's reward is a weighted sum of its local performance and the global performance of the entire corridor.

$$
\text{reward} = (0.7 \times \text{localReward}) + (0.3 \times \text{globalReward})
$$

| Component | Basis | Goal |
| :--- | :--- | :--- |
| **Local Reward (70%)** | Minimizing stopped cars in the agent's own detector zones. | Ensures local efficiency. |
| **Global Reward (30%)** | Minimizing the average number of stopped cars across *all* detector zones in the corridor. | Encourages cooperation and corridor-wide optimization. |


