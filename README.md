# IoTWarden: A Deep Reinforcement Learning based Real-time Defense System to Mitigate Trigger-action IoT Attacks 
This repository contains the Python implementation of _IoTWarden_, a **Deep Reinforcement Learning (DRL)** based realtime defense system that helps a security agent in an IoT network take essential defense actions against an ongoing remote injection (trigger-action) attack. IoTWarden solves a **Markov Decision Process (MDP)** using **Deep Q-Network (DQN)**[1] algorithm to determine the optimal defense policy (sequence of actions) for the security agent.

Besides, IoTWarden also discovers temporal patterns of different IoT events recorded in a dataset named PEEVES[2] using an **LSTM**[3] based **Recurrent Neural Network (RNN)** and generate a set of likely event sequenes for a rational attacker.

# Dataset
**Dataset Name**: Peeves: Physical Event Verification in Smart Homes[2]

**Link**: https://ora.ox.ac.uk/objects/uuid:75726ff7-fee1-420d-8a17-de9572324c7d

**About the dataset**: https://ora.ox.ac.uk/objects/uuid:75726ff7-fee1-420d-8a17-de9572324c7d/download_file?file_format=pdf&safe_filename=readme.pdf&type_of_work=Dataset).

# Codebase Information
**Language used**: Python

**ML Libraries used**: TensorFlow, OpenAI Gym, Scikit-Learn

**Libraries used for visualization**: Matplotlib

**Other Libraries used**: Numpy, pandas

# Usage
1. Install the following dependencies: `TensorFlow`, `OpenAI Gym`, `Python`, `NumPy`, `pandas`, and `Matplotlib`.
2. Simply execute the `main.py` script to run IoTWarden.
3. To extract optimal event sequences for the attacker, execute the notebook named `sequence_modeling_LSTM.ipynb`.

# Ouput
1) Output for optimal event sequences extraction:
![Training and validation accuracy of the LSTM-based RNN](Evaluation/accuracy_plot.pdf)

2) 

# Reference
[1] Mnih, Volodymyr & Kavukcuoglu, Koray & Silver, David & Graves, Alex & Antonoglou, Ioannis & Wierstra, Daan & Riedmiller, Martin. (2013). Playing Atari with Deep Reinforcement Learning. 
[2] Birnbach, S., & Eberz, S. (2019). Peeves: Physical Event Verification in Smart Homes. University of Oxford.
[3] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, p. 1735–1780, nov 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735
