import os
import pickle
import numpy as np

import torch


class RandomSpawnEvaluation(object):
    def __init__(self, network, agent, environment, total_episodes, max_steps_per_ep,
                 is_exp=True, is_aleatoric=False, is_epistemic=False, seed=0):
        self.network = network
        self.agent = agent
        self.environment = environment
        self.total_episodes = total_episodes
        self.max_steps_per_ep = max_steps_per_ep
        self.set_seed(seed)
        self.is_exp = is_exp
        self.is_aleatoric = is_aleatoric
        self.is_epistemic = is_epistemic

        # vars for persisting transitions
        self.sample_loss_map = [[[] for _ in range(10)] for _ in range(10)]
        self.exp_loss_map = [[[] for _ in range(10)] for _ in range(10)] if is_exp else None
        self.aleatoric_map = [[[] for _ in range(10)] for _ in range(10)] if is_aleatoric else None
        self.epistemic_map = [[[] for _ in range(10)] for _ in range(10)] if is_epistemic else None

    def run(self):
        episode_count = 0
        while episode_count < self.total_episodes:
            self.run_episode()
            episode_count += 1
        self.sample_loss_map = self.mean_map(self.sample_loss_map)
        if self.is_exp:
            self.exp_loss_map = self.mean_map(self.exp_loss_map)
        if self.is_aleatoric:
            self.aleatoric_map = self.mean_map(self.aleatoric_map)
        if self.is_epistemic:
            self.epistemic_map = self.mean_map(self.epistemic_map)

    def run_episode(self):
        s = self.environment.start(self.sample_xy())
        a = self.agent.start(s)
        done = False
        step_count = 0
        transitions = []
        while not (done or step_count == self.max_steps_per_ep):
            ns = self.environment.step(a)
            step_count += 1
            self.agent.update(s, a, ns, done)
            self.populate_maps(s, a, ns)
            a = self.agent.get_action(ns)
            s = ns
        return transitions

    def populate_maps(self, s, a, ns):
        delta = self.network.predict_mean(self.network.encode_state(s),
                                          self.network.encode_action(torch.LongTensor([[a]])))
        delta = delta.data.cpu().numpy()
        x, y = self.get_index(s)
        loss = np.sum(np.abs(ns - (s + delta)))
        self.sample_loss_map[x][y].append(loss)

        if self.is_exp:
            exp_ns = self.environment.expected_step(s, a)
            exp_loss = np.sum(np.abs(exp_ns - (s + delta)))
            self.exp_loss_map[x][y].append(exp_loss)

        if self.is_aleatoric:
            var = self.network.predict_aleatoric_variance(self.network.encode_state(s),
                                                          self.network.encode_action(torch.LongTensor([[a]])))
            self.aleatoric_map[x][y].append(var)

        if self.is_epistemic:
            var = self.network.predict_epistemic_variance(self.network.encode_state(s),
                                                          self.network.encode_action(torch.LongTensor([[a]])))
            self.epistemic_map[x][y].append(var)

    def persist_maps(self, logdir):
        with open(os.path.join(logdir, "sample_loss"), "wb") as f:
            pickle.dump(self.sample_loss_map, f)
        if self.is_exp:
            with open(os.path.join(logdir, "exp_loss"), "wb") as f:
                pickle.dump(self.exp_loss_map, f)
        if self.is_aleatoric:
            with open(os.path.join(logdir, "aleatoric"), "wb") as f:
                pickle.dump(self.aleatoric_map, f)
        if self.is_epistemic:
            with open(os.path.join(logdir, "epistemic"), "wb") as f:
                pickle.dump(self.epistemic_map, f)

    def sample_xy(self):
        x = np.random.uniform()
        y = np.random.uniform()
        return x, y

    def set_seed(self, seed):
        if seed != -1:
            np.random.seed(seed)

    def get_index(self, state):
        # TODO: Might need to make discretization flexible. Right now it consists of a 10x10 grid in the unit square
        x, y = state
        return min(max(0, int(x / 0.1)), 9), min(max(0, int(y / 0.1)), 9)

    def mean_map(self, loss_map):
        mean = np.zeros((10, 10))
        for p in range(10):
            for q in range(10):
                mean[p][q] = np.mean(loss_map[p][q])
        return mean
