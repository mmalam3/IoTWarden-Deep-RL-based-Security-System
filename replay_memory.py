import random
from collections import deque, namedtuple


class ReplayMemory(object):
    def __init__(self, memory_size):
        super(ReplayMemory, self).__init__()
        self.memory = deque([], maxlen=memory_size)

    def add_experience(self, *args):
        # create a new tuple Transition = ('state', 'action', 'next_state', 'reward')
        # that will be constituted from the return value of step function of the env
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory.append(Transition(*args))

    def sample_experience(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)