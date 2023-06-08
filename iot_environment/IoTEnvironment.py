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
import random


class IoTEnv(gym.Env):
    def __init__(self):
        # Initialize the environment
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(12)

        self.state = [0] * 12
        self._latest_state = 0
        self._latest_state_pool = 0
        self._goal_state = 5 # Goal state: window
        self.done = False

        self.num_injection = 0 # action: 0
        self.individual_injection_reward = 5
        # self.silent_period = 0
        # self.silent_threshold = 3 # continuous silent period
        self.state_space_size = len(self.state)

        self.num_defense_actions = 0
        self.individual_defense_reward = 10

        # self.discount_factor = 0.95
        self.state_pools = [[0],
                            [1, 2, 3, 4],
                            [5, 6],
                            [7, 8],
                            [9, 10, 11]]

    def check_goal_status(self):
        if self.state[self._goal_state] == 1: return True
        else: return False

    def reset(self):
        # Reset the environment to its initial state and
        # return the initial observation state at the start of the game
        self.state = [0] * 12
        self._latest_state = 0
        self._latest_state_pool = 0
        self.done = False
        self.num_injection = 0  # action: 0
        # self.silent_period = 0
        self.num_defense_actions = 0

        return self.state, {}

    def step(self, action):
        curr_states = self.state_pools[self._latest_state_pool]
        injection_threshold = 0.3
        attack_gain = 0
        defense_gain = 0

        # attacker's action: 'inject_fake_events'
        if action == 0:
            self.num_injection += 1
            if (self.num_injection / self.state_space_size) >= injection_threshold:
                # attacker aggression in dangerous mode
                attack_gain -= int(injection_threshold * self.num_injection) # defender's POV
                # attack_gain += int(injection_threshold * self.num_injection) # attacker's POV

            if self._latest_state_pool == len(self.state_pools) - 2:  # 2nd last pool
                if self.state[self._goal_state] == 1:
                    attack_gain += 100
                    self.done = True
                    self.reset()

            elif curr_states[-1] == self._latest_state:  # last state of the current pool
                self._latest_state_pool += 1
                curr_states = self.state_pools[self._latest_state_pool]  # update curr_states
                attack_gain += self.individual_injection_reward + (self._latest_state_pool * self.num_injection)

        # attacker's action: 'monitor_defense_actions'
        elif action == 1:
            # demotivate attacker to monitor defense actions
            attack_gain += self.num_defense_actions

        # defender's action: 'monitor_attack_actions'
        elif action == 2:
            defense_gain += self.num_injection # defender's POV
            # defense_gain -= self.num_injection # attacker's POV

        # defender's action: blocking trigger actions
        elif action == 3:
            self.num_defense_actions += 1
            defense_gain += self.individual_defense_reward
            defense_gain += (self._latest_state_pool * self.num_defense_actions)

            if self._latest_state_pool >= 1:
                self._latest_state_pool -= 1

        # choose the index of first zero element
        for _state in curr_states:
            if self.state[_state] == 0:
                self._latest_state = _state
                break

        self.state[self._latest_state] = 1

        # calculate reward
        reward = defense_gain - attack_gain # defender POV
        # reward = attack_gain - defense_gain # attacker's POV

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
