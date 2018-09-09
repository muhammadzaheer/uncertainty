import numpy as np


class Agent(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action(self, s):
        return np.random.randint(0, self.num_actions)

    def update(self, s, a, ns, done):
        pass

    def start(self, s):
        return self.get_action(s)
