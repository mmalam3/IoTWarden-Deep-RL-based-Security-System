# IoTWarden: A Deep Reinforcement Learning based Real-time Defense System to Mitigate Trigger-action IoT Attacks 
This repository contains the Python implementation of _IoTWarden_, a **Deep Reinforcement Learning (DRL)** based realtime defense system that helps a security agent in an IoT network take essential defense actions against an ongoing remote injection (trigger-action) attack. IoTWarden solves a **Markov Decision Process (MDP)**[4] using **Deep Q-Network (DQN)**[1] algorithm to determine the optimal defense policy (sequence of actions) for the security agent.

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
1) **Security agent's reward**: Since the goal of an MDP is always maximizing total reward achievable over a limited number of epochs in a single game iteration, IoTWarden always tries to choose a defense policy that lets the security agent achieves maximized security rewards, similar to the shown in the following figure:
![Defense agent's reward over 250 epochs](https://github.com/mmalam3/DQN-TensorFlow-Gym/blob/main/Evaluation/reward_vs_episodes.png)

2) **Performance evaluation of the LSTM-based RNN**: The following figures shows the training and validation accuracy and loss of the LSTM-based RNN used to extract the temporal dependencies of the dataset events and generate optimal sequences of events for the attacker. 
![Training and validation accuracy of the LSTM-based RNN](https://github.com/mmalam3/DQN-TensorFlow-Gym/blob/main/Evaluation/accuracy_plot.png)
![Training and validation loss of the LSTM-based RNN](https://github.com/mmalam3/DQN-TensorFlow-Gym/blob/main/Evaluation/loss_plot.png) 

# Published Research Paper
**Pre-print:** [IoTWarden_Alam_2024.pdf](https://github.com/mmalam3/DQN-TensorFlow-Gym/blob/7fddff858006664f841438be3908deb93bcad6ed/IoTWarden_Alam_2024.pdf)

# Reference
[1] Mnih, Volodymyr & Kavukcuoglu, Koray & Silver, David & Graves, Alex & Antonoglou, Ioannis & Wierstra, Daan & Riedmiller, Martin. (2013). Playing Atari with Deep Reinforcement Learning. 

[2] Birnbach, S., & Eberz, S. (2019). Peeves: Physical Event Verification in Smart Homes. University of Oxford.

[3] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, p. 1735–1780, nov 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735.

[4] R. Bellman, “A markovian decision process,” Indiana Univ. Math. J., vol. 6, pp. 679–684, 1957.
