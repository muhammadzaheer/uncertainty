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


class NoiseWorldv0(object):
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

    def sample_start_state(self):
        x = np.random.uniform()
        y = np.random.uniform()
        return x, y


class NoiseWorldv1(NoiseWorldv0):
    # A unit square where the displacement has zero mean gaussian noise
    def __init__(self):
        super(NoiseWorldv1, self).__init__()
        self.noise_variance = [[0.000025, 0.0], [0.0, 0.000025]]  # i.e. std: 0.05 for each dimension

    def step(self, action):
        x, y = self.current_state
        if x > 0.9 and y > 0.9:
            # Non-Markov region (considerably smaller than NoiseWorldv0)
            noise = np.random.multivariate_normal(self.noise_mean, self.noise_variance, 1)[0]
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0] + noise[0]))
            ny = min(self.max_y, max(0.0, y + _action[1] + noise[1]))
        else:
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0]))
            ny = min(self.max_y, max(0.0, y + _action[1]))
        ns = (nx, ny)
        self.current_state = np.asarray(ns)
        self.previous_action = action

        return np.copy(self.current_state)


class NoiseWorldv2(NoiseWorldv0):
    # A unit square where the displacement has zero mean gaussian noise
    def __init__(self):
        super(NoiseWorldv2, self).__init__()
        self.noise_variance = [[0.000025, 0.0], [0.0, 0.000025]]  # i.e. std: 0.05 for each dimension

    def step(self, action):
        x, y = self.current_state
        if x > 0.9 and y < 0.1:
            # Non-Markov region (considerably smaller than NoiseWorldv0)
            noise = np.random.multivariate_normal(self.noise_mean, self.noise_variance, 1)[0]
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0] + noise[0]))
            ny = min(self.max_y, max(0.0, y + _action[1] + noise[1]))
        else:
            _action = self.action_set[action]
            nx = min(self.max_x, max(0.0, x + _action[0]))
            ny = min(self.max_y, max(0.0, y + _action[1]))
        ns = (nx, ny)
        self.current_state = np.asarray(ns)
        self.previous_action = action

        return np.copy(self.current_state)


class Sinev0(object):
    # A unit square where the displacement has zero mean gaussian noise
    def __init__(self):
        self.alpha = 4
        self.beta = 13
        self.noise_region = (0.4, 0.6)
        self.noise = 0.1

        self.current_state = np.array([np.random.uniform(low=-1.0, high=2.0)])

    def start(self, start_state=None):
        if start_state is not None:
            self.current_state = start_state
        return np.copy(self.current_state)

    def step(self, action):
        x = self.current_state[0]
        mu = 0.0
        low, high = self.noise_region
        if low < x < high:
            sigma = self.noise
        else:
            sigma = 0.0

        w = np.random.normal(mu, sigma)
        y = x + np.sin(self.alpha*x) + np.sin(self.beta * x) + w

        return np.array([y])

    def expected_step(self, state, action):
        x = state[0]
        mu = 0.0
        sigma = 0.0
        w = np.random.normal(mu, sigma)
        y = x + np.sin(self.alpha * (x + w)) + np.sin(self.beta * (x + w)) + w
        return np.array([y])

    def sample_state(self):
        return np.array([np.random.uniform(low=-1.0, high=2.0)])

    def domain(self):
        return -1.0, 2.0

    def mean_function(self, num=300):
        # Plotting the mean line
        x = np.linspace(-1, 2, num=num)
        y = x + np.sin(self.alpha * x) + np.sin(self.beta * x)
        return x, y

    def generate_samples(self, num=300):
        samples = []
        for k in range(num):
            x = np.random.uniform(low=-1.0, high=2.0)
            self.current_state[0] = x
            y = self.step(0)[0]
            samples.append((x, y))
        samples = np.array(samples)
        x, y = samples[:, 0], samples[:, 1]
        return x, y

    def plot_samples(self, ax, out_path=None):

        # Plotting 900 random samples from (-1, 2.0)
        samples = []
        for k in range(900):
            x = np.random.uniform(low=-1.0, high=2.0)
            y = self.sample_y(x)
            samples.append((x, y))
        samples = np.array(samples)
        ax.scatter(samples[:, 0], samples[:, 1], c=sns.xkcd_rgb["light pink"], alpha=0.6, marker='.')

        # Plotting the mean line
        x = np.linspace(-1, 2, num=300)
        y = x + np.sin(self.alpha * x) + np.sin(self.beta * x)
        ax.plot(x, y, sns.xkcd_rgb["black"], lw=1)

        # Plotting the samples in the dataset
        d = np.array(self.samples)
        ax.scatter(d[:, 0], d[:, 1], c=sns.xkcd_rgb["sea blue"])

        if out_path is not None:
            ax.savefig(os.path.join(out_path, 'sine.png'))

if __name__ == '__main__':
    env = NoiseWorldv0()
    env.start()
    env.step(0)
