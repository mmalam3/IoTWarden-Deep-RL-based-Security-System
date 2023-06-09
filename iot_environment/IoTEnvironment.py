'''
Optimal sequence predicted through LSTM:
[dooron] -> [fanon] -> [scron -> camon -> pcon] -> [lighton] -> [radon] ->
[shdup -> rang -> win2on] -> [fridgeon -> coffee3]

Action map: {0: 'inject_fake_events', 1: 'monitor_defense_actions',
            2: 'monitor_attack_actions', 3: 'block_trigger_action'}

Note: We are implementing it from the defense perspective
'''


import gym
from gym import spaces


class IoTEnv(gym.Env):
    def __init__(self):
        # Initialize the environment
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(12)

        self.state = [0] * 12
        self.state_space_size = len(self.state)
        self._latest_state = 0

        self._goal_state = 9 # Goal state: window
        self.done = False

        self.n_i = 0 # action: 3 (num_injection_action)
        self.n_z = 1 # action: 1 (num_monitor_defense_action)
        self.n_b = 0 # action: 3 (num_defense_actions (block))
        self.n_m = 1 # action: 2 (num_monitor_attack_action)

        self.attack_proximity_factor = 0
        self.proximity_threshold = 0.5

    def reset(self):
        # Reset the environment to its initial state and
        # return the initial observation state at the start of the game
        self.state = [0] * 12
        self._latest_state = 0
        self.done = False
        self.n_i = 0  # action: 3
        self.n_z = 1  # action: 1
        self.n_b = 0  # action: 3
        self.n_m = 1  # action: 2
        self.attack_proximity_factor = 0

        return self.state, {}

    def step(self, action):
        injection_threshold = 0.8

        G_r = 0 # goal_node_reward
        r_i = 0 # reward_for_injection
        r_z = 1 # reward_for_checking_defense_action
        r_b = 0 # reward_for_block_ops
        r_m = 1 # reward_for_monitoring_attack_action

        if action == 0: # attacker's action: 'inject_fake_events'
            r_i += 5

            self.n_i += 1
            self._latest_state += 1
            self.state[self._latest_state] = 1
            self.attack_proximity_factor = self._latest_state / self.state_space_size

            # check whether goal node is compromised
            if self.state[self._goal_state] == 1:
                G_r += 50
                self.done = True
                self.reset()

        elif action == 1: # attacker's action: 'monitor_defense_actions'
            r_z += 0.5
            self.n_z += 1

        elif action == 2: # defender's action: 'monitor_attack_actions'
            r_m += 1
            self.n_m += 1

        elif action == 3: # defender's action: blocking trigger actions
            r_b += 10

            self.n_b += 1
            self.state[self._latest_state] = 0

            # only push the attack backward if the current node is not the very first node
            if self._latest_state > 0:
                self._latest_state -= 1

            self.attack_proximity_factor = self._latest_state / self.state_space_size

        # calculate reward based on the reward function
        r_attack = (self.n_i * r_i * self.attack_proximity_factor) / (self.n_i * r_i + self.n_z * r_z) + G_r

        condition = (self.n_i * self.attack_proximity_factor) / (self.n_i + self.n_z)

        if condition < injection_threshold:
            r_defense = self.n_m * r_m
        else:
            r_defense = (self.n_b * r_b) / (self.n_b + self.n_m)

        reward = r_defense - r_attack

        # x = reward_for_current_action # x = r_d + r_ma + r_md - r_i
        #
        # y_1 = (self.n_i + self.n_z) - (self.n_b + self.n_m)
        # y_2 = (self.attack_proximity_factor - 1)
        # y= y_1 * y_2
        #
        # z = self.n_i * injection_threshold
        #
        # reward = x + y - z - goal_node_reward

        # specify returning values
        obs = self.state
        info = {}
        status = self.done

        return obs, reward, status, info


# Register the environment with OpenAI Gym
gym.envs.register(
    id='IoTEnv',
    entry_point='IoTEnvironment:IoTEnv'
)


# create the environment
def create_environment():
    env = gym.make('IoTEnv')

    print(f'State space:\n {list(range(env.observation_space.n))}')
    print(f'Action space:\n {list(range(env.action_space.n))}')

    return env
