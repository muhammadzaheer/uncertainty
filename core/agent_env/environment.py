import numpy as np


class AccelerateWorldv1(object):
    # A unit square where the displacement increases by a 'doubling factor'
    # on repeated actions in the same direction
    def __init__(self, doubling_factor=1.5):
        # Start state and no of rows, cols in unit grid
        self.max_x = 1.0
        self.max_y = 1.0

        self.start_state = (0.75, 0.5)

        # Possible default actions in tabular environment
        self.action_set = [(0, 0.01), (0, -0.01), (0.01, 0), (-0.01, 0)]  # R, L, U, D
        self.max_actions = len(self.action_set)

        self.current_state = None
        self.previous_action = 0
        # if action == previous_action && non_markov_region: double the movement factor
        self.movement_factor = 1
        self.doubling_factor = doubling_factor

    def start(self):

        self.current_state = np.asarray(self.start_state)
        # Returning a copy of the current state
        return np.copy(self.current_state)

    def step(self, action):
        x, y = self.current_state
        if x > 0.5:
            # Non-Markov region
            if action == self.previous_action:
                self.movement_factor *= self.doubling_factor

            else:
                self.movement_factor = 1
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0]*self.movement_factor))
            ny = min(self.max_y, max(0.0, y + _action[1]*self.movement_factor))
        else:
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0]))
            ny = min(self.max_y, max(0.0, y + _action[1]))
            self.movement_factor = 1.0
        ns = (nx, ny)
        self.current_state = np.asarray(ns)
        self.previous_action = action

        return np.copy(self.current_state)

    def expected_step(self, action):
        raise NotImplementedError


class NoiseWorld(object):
    # A unit square where the displacement has zero mean gaussian noise
    def __init__(self):
        # Start state and no of rows, cols in unit grid
        self.max_x = 1.0
        self.max_y = 1.0

        self.start_state = (0.75, 0.5)

        # Possible default actions in tabular environment
        self.action_set = [(0, 0.01), (0, -0.01), (0.01, 0), (-0.01, 0)]  # R, L, U, D
        self.max_actions = len(self.action_set)

        self.current_state = None
        self.previous_action = 0
        
        self.noise_mean = [0, 0]
        self.noise_variance = [[0.0001, 0.0], [0.0, 0.0001]] # i.e. std: 0.01 for each dimension
        # self.noise_variance = [[0.0025, 0.0], [0.0, 0.0025]] # i.e. std: 0.05 for each dimension
        # self.noise_variance = [ [0.0, 0.0], [0.0, 0.0]]

    def start(self, start_state=None):
        if start_state is None:
            self.current_state = np.asarray(self.start_state)
        else:
            self.current_state = np.asarray(start_state)
        return np.copy(self.current_state)

    def step(self, action):
        x, y = self.current_state
        if x > 0.5:
            # Non-Markov region
            noise = np.random.multivariate_normal(self.noise_mean, self.noise_variance, 1)[0]
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0]+noise[0]))
            ny = min(self.max_y, max(0.0, y + _action[1]+noise[1]))
        else:
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0]))
            ny = min(self.max_y, max(0.0, y + _action[1]))
        ns = (nx, ny)
        self.current_state = np.asarray(ns)
        self.previous_action = action

        return np.copy(self.current_state)

    def expected_step(self, state, action):
        # Calculates the expected next state
        x, y = state
        action = self.action_set[action]
        # Getting the coordinate representation of the state
        nx = x + action[0]
        ny = y + action[1]
        ns = (nx, ny)
        return np.asarray(ns)


if __name__ == '__main__':
    env = NoiseWorld()
    env.start()
    env.step(0)
