import random
import math
from collections import namedtuple, deque
from itertools import count
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(0)

# import necessary modules
from dqn import DQN
from replay_memory import ReplayMemory

step_done = 0


class Agent(object):
    def __init__(self, arg):
        super(Agent, self).__init__()
        self.arg = arg
        self.batch_size = 16
        self.discount_factor = 0.95
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 100000
        self.tau = 0.05
        self.lr = 1e-5
        self.n_actions = 4
        self.n_observations = 20
        # self.writer = tf.summary.create_file_writer('logs/')
        self.episodic_loss = 0

        self.episode_returns = []

        # create network
        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.set_weights(self.policy_net.get_weights())

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr, amsgrad=True)
        self.memory = ReplayMemory(20000)

    # method to select action using policy_net (DQN)
    def select_action(self,state):
        global step_done
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step_done / self.eps_decay)
        step_done += 1

        if sample > eps_threshold:
            # get q_values from the policy network (DQN)
            q_values = self.policy_net(state)

            # retrieve the acton producing max of q_values
            action = tf.argmax(q_values, axis=1).numpy()[0]
            return tf.convert_to_tensor([[action]], dtype=tf.int64)

        else:
            return tf.convert_to_tensor([[np.random.choice(self.n_actions)]], dtype=tf.int64)

    def learn_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sets the gradients of all optimized Tensor to zero and calculate loss
        with tf.GradientTape() as tape:
            # sample a random experience from the replay buffer
            transitions = self.memory.sample_experience(self.batch_size)

            Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
            batch = Transition(*zip(*transitions))

            non_final_mask = tf.convert_to_tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=tf.bool)
            non_final_next_states = tf.concat([s for s in batch.next_state if s is not None], axis=0)

            state_batch = tf.concat(batch.state, axis=0)
            action_batch = tf.concat(batch.action, axis=0)
            reward_batch = tf.concat(batch.reward, axis=0)

            state_action_values = tf.gather_nd(self.policy_net(state_batch),
                                               tf.stack([tf.range(self.batch_size), action_batch], axis=1))

            next_state_values = tf.zeros(self.batch_size, dtype=tf.float32)

            # get the indices (in vector form, not in scalar form) to be updated
            indices = tf.where(non_final_mask)

            # get the indices scalar and convert the list into a tensor
            indices_scalar = tf.constant([index.numpy()[0] for index in indices])

            # define the updates to be written
            updates = np.max(self.target_net(non_final_next_states), axis=1)

            # apply the updates using tf.tensor_scatter_nd_update
            next_state_values = tf.tensor_scatter_nd_update(next_state_values,
                                                            np.expand_dims(indices_scalar, axis=1),
                                                            updates)

            expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

            # choose the Huber loss function as the criterion
            criterion = tf.keras.losses.Huber()

            # compute the Huber loss (note: the output 'loss' is a tensor, not variable)
            loss = criterion(state_action_values, tf.expand_dims(expected_state_action_values, axis=1))

            # self.episodic_loss += loss
            self.episodic_loss += loss.numpy()

            # Backpropagation
            # gradients = tape.gradient(loss, self.policy_net.trainable_variables)
            gradients = tape.gradient(loss, self.policy_net.trainable_variables)
            # print(f'Gradients: \n {gradients}')

            self.optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

    # method to update target network
    def updateTargetNetwork(self):
        target_net_weights = self.target_net.get_weights()
        policy_net_weights = self.policy_net.get_weights()
        new_weights = []
        for target_w, policy_w in zip(target_net_weights, policy_net_weights):
            new_w = self.tau * policy_w + (1 - self.tau) * target_w
            new_weights.append(new_w)
        self.target_net.set_weights(new_weights)

    def plot_rewards(self, show_result=False):
        plt.figure(1)
        rewards = np.array(self.episode_returns, dtype=np.float32)
        if show_result:
            plt.title('Result')
        else:
            # plt.clf()
            plt.title('Training...')
        plt.xlabel('Epochs')
        plt.ylabel('Returns')
        plt.plot(rewards, 'red')
        plt.savefig("training_reward_test.png")

    # method to train the agent
    def train_agent(self, env):
        for e in range(1000):
            state, info = env.reset()
            current_return = 0

            state = tf.convert_to_tensor(np.array(state).reshape(1, 20), dtype=tf.float32)

            for t in range(200):
                # select an action for the current state
                action = self.select_action(state)
                action = tf.cast(action, tf.int32).numpy()[0]

                # make a transition by calling the step() of env
                # output of step() -> (obs, reward, status, info)
                observation, reward, terminated, _ = env.step(action)
                observation = np.array(observation).reshape(1, 20)

                current_return += reward

                # convert current reward into a scalar tensor
                reward = tf.convert_to_tensor([reward], dtype=tf.float32)

                # status of the game
                done = terminated

                if terminated: # goal node has been compromised by the attacker
                    next_state = None
                else:
                    # find next_step and convert into a scalar tensor
                    # next_state = tf.constant(observation, dtype=tf.float32)[None, :]
                    next_state = tf.convert_to_tensor(observation, dtype=tf.float32)

                # add experience to the replay memory
                self.memory.add_experience(state, action, next_state, reward)

                # print(f'\nCurrent state: \n{state}')
                # print(f'\nNext state: \n{next_state}')

                state = next_state

                # call the learn model
                self.learn_model()

                # update target network
                self.updateTargetNetwork()

                # if (e + 1) % 2 == 0:
                #     tf.saved_model.save(self.policy_net.state_dict(), "models/model_test.pth")

                if done or t == 199:
                    self.episode_returns.append(current_return)
                    # print(f'Current return: {current_return}')
                    self.plot_rewards()
                    # env.closeEnvConnection()
                    env.reset()
                    print(f'Episodes:{e + 1}, Timestep: {t}, Reward: {current_return}')
                    break

            # self.writer.add_scalar("Loss/train", self.episodic_loss, (e + 1))
            # self.writer.add_scalar("Reward/Train", current_return, (e + 1))
            # self.writer.flush()
            self.episodic_loss = 0.0

        self.plot_rewards(show_result=True)
        plt.show()