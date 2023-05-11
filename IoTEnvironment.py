'''
State map: {0: 'camera', 1: 'fridge', 2: 'pc', 3: 'radiation_sensor', 4: 'fan', 5: 'screen',
6: 'window_shade', 7: 'ring_bell', 8: 'door', 9: 'window', 10: 'coffee_machine', 11: 'light'}

Action map: {0: 'inject_fake_events', 1: 'monitor_defense_actions', 2: 'block_trigger_action',
3: 'monitor_state_changes'}
'''


import gym
from gym import spaces
import random


class IoTEnv(gym.Env):
    def __init__(self):
        # Initialize the environment
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(12)

        # print(f'Action Space:\n {self.action_space}')
        # print(f'State Space:\n {self.observation_space}')

        self.state = [0] * 12
        self._latest_state = 0
        self._latest_state_pool = 0
        self._goal_state = 9 # Goal state: window
        self.done = False

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

        return self.state, {}
        # return ts.restart(np.array([self._state], dtype=np.int32))

    def step(self, action):
        # Take an action and return the new observation, reward,
        # and whether the episode is done or not
        if self.done:
            return self.reset()

        reward = self.get_reward(action)  # Get the reward
        next_state = self.get_next_state(action)  # Update state
        self.state[next_state] = 1

        # transitions
        if self.check_goal_status():  # _episode_ended flag is TRUE
            self.done = True
            reward -= 100

        # specify returning values
        obs = self.state
        info = {}
        status = self.done

        return obs, reward, status, info

    def get_reward(self, action):
        reward = 0  # cost for the defense; defender tries to minimize it

        if action == 0:  # attacker's action: 'inject_fake_events'
            reward -= 50
        elif action == 1:  # attacker's action: 'monitor_defense_actions'
            reward -= 5
        elif action == 3:  # defender's action: 'monitor_state_changes'
            reward -= 1
        else:  # defender's action: blocking trigger actions
            reward += 55

        return reward

    def get_next_state(self, action):
        curr_states = self.state_pools[self._latest_state_pool]

        if action == 0:  # attacker's action: 'inject_fake_events'
            if self._latest_state_pool == len(self.state_pools) - 1:  # last pool
                if self.state[self._goal_state] == 1:
                    # self.done = True
                    self.reset()

            elif curr_states[-1] == self._latest_state: # last state of the current pool
                self._latest_state_pool += 1
                curr_states = self.state_pools[self._latest_state_pool]  # update curr_states

        # choose a state from curr_states randomly
        self._latest_state = random.choice(curr_states)

        return self._latest_state

    # def render(self, mode='human'):
    #     # Render the environment


# Register the environment with OpenAI Gym
gym.envs.register(
    id='IoTEnv-v0',
    entry_point='IoTEnvironment:IoTEnv'
)


# create the environment
def create_environment():
    env = gym.make('IoTEnv-v0')

    print(f'State space:\n {list(range(env.observation_space.n))}')
    print(f'Action space:\n {list(range(env.action_space.n))}')

    return env
