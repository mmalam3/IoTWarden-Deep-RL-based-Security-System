# include paths for modules to import
import sys
sys.path.append('./network/')

# import necessary modules
import random
import math
from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import tensorflow as tf
tf.random.set_seed(40)

# import DQN and ReplayMemory classes
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
        self.eps_end = 0.1
        self.eps_decay = 100000
        self.tau = 0.05
        self.lr = 1e-3
        self.n_actions = 4
        self.n_observations = 12
        self.n_episodes = [215]
        self.n_epochs = 1
        self.timesteps = 100
        self.episodic_loss = 0
        self.episode_returns = []

        self.writer = tf.summary.create_file_writer('logs')
        # self.writer_action_ratio = tf.summary.create_file_writer('logs/action_ratio')
        # self.writer_injections = tf.summary.create_file_writer('logs/injections')
        # self.writer_block_ops = tf.summary.create_file_writer('logs/block_ops')

        # create network
        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.set_weights(self.policy_net.get_weights())

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr, amsgrad=True)
        self.memory = ReplayMemory(40000)

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
    def update_target_network(self):
        target_net_weights = self.target_net.get_weights()
        policy_net_weights = self.policy_net.get_weights()
        new_weights = []
        for target_w, policy_w in zip(target_net_weights, policy_net_weights):
            new_w = self.tau * policy_w + (1 - self.tau) * target_w
            new_weights.append(new_w)
        self.target_net.set_weights(new_weights)

    def train_agent(self, env):
        for num_episode in self.n_episodes:
            # print(f'\n\nCurrent episode size: {num_episode}\n')
            total_curr_attack = 0

            # for episode in range(self.n_episodes):
            for episode in range(num_episode):
                episode_return_total = 0

                # count number of event injection actions
                num_event_injections = 0
                num_block_trigger_action = 0
                total_actions = 0

                for epoch in range(self.n_epochs):
                    # start the timer
                    t1_start = perf_counter()

                    state, info = env.reset()
                    current_return = 0
                    state = tf.convert_to_tensor(np.array(state).reshape(1, 12), dtype=tf.float32)

                    for t in range(self.timesteps):
                        # select an action for the current state
                        action = self.select_action(state)
                        action = tf.cast(action, tf.int32).numpy()[0]

                        # check whether the action is "injection" or "block"
                        if action == 0:
                            num_event_injections += 1
                        elif action == 3:
                            num_block_trigger_action += 1

                        # increase the count of total actions taken so far
                        total_actions += 1

                        # make a transition by calling the step() of env
                        # output of step() -> (obs, reward, status, info)
                        observation, reward, terminated, _ = env.step(action)
                        observation = np.array(observation).reshape(1, 12)

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

                        state = next_state

                        # call the learn model
                        self.learn_model()

                        # update target network
                        self.update_target_network()

                        # if done or t == 99:
                        if done or t == (self.timesteps - 1):
                            if done:
                                print('\nGoal node compromised')
                                total_curr_attack += 1

                            # self.episode_returns.append(current_return)
                            episode_return_total += current_return # accumulate episode return for current epoch

                            # self.plot_rewards()
                            env.reset()
                            # print(f'Episode:{episode+1}, Epoch: {epoch+1}, Timestep: {t}, Reward: {current_return}')

                            # Stop the stopwatch / counter and calculate time overhead
                            time_overhead = perf_counter() - t1_start

                            # # flush event_injections and block_ops count in separate logs
                            # # using different writers, but same name

                            time_to_compromise = perf_counter() - t1_start

                            with self.writer.as_default():
                                # time over head vs episodes
                                tf.summary.scalar(name='time-to-compromise-goal-node', data=time_to_compromise, step=(episode + 1))
                                self.writer.flush()

                            # with self.writer_injections.as_default():
                            #     # time over head vs episodes
                            #     tf.summary.scalar(name='injection_block_ops_vs_episode', data=num_event_injections, step=(episode + 1))
                            #     self.writer_injections.flush()
                            #
                            # with self.writer_block_ops.as_default():
                            #     tf.summary.scalar(name='injection_block_ops_vs_episode', data=num_block_trigger_action, step=(episode + 1))
                            #     self.writer_block_ops.flush()

                            # with self.writer_action_ratio.as_default():
                            #     tf.summary.scalar(name='blocking_injection_action_ratio',
                            #                       data=(num_block_trigger_action / num_event_injections),
                            #                       step=(episode + 1))
                            #
                            #     tf.summary.scalar(name='injection_total_actions_ratio',
                            #                       data=(num_event_injections / total_actions),
                            #                       step=(episode + 1))
                            #
                            #     tf.summary.scalar(name='injection_total_actions_ratio',
                            #                       data=(num_event_injections / total_actions),
                            #                       step=(episode + 1))
                            #
                            #     self.writer_action_ratio.flush()

                            break

                    self.episodic_loss = 0.0

                # after all epochs, calculate avg_episode_return
                avg_episode_return = episode_return_total / self.n_epochs
                self.episode_returns.append(avg_episode_return)
                # self.plot_rewards()
                print(f'Episode:{episode+1}, Avg_reward: {avg_episode_return}')

                # flush returns for each episode in Tensorboard
                # with self.writer.as_default():
                #     tf.summary.scalar(name='reward', data=avg_episode_return, step=(episode+1))
                #     self.writer.flush()

