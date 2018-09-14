import numpy as np
import os
import pickle


class RandomSpawnExperiment(object):
    def __init__(self, agent, environment, total_episodes, max_steps_per_ep, persist=False, persist_dir=None, seed=0):
        self.agent = agent
        self.environment = environment
        self.total_episodes = total_episodes
        self.max_steps_per_ep = max_steps_per_ep
        self.set_seed(seed)

        # vars for persisting transitions
        self.persist = persist
        self.persist_dir = persist_dir
        self.persist_interval = 25
        self.transition_count = 0

    def run(self):
        episode_count = 0
        transitions = []
        while episode_count < self.total_episodes:
            ep_transitions = self.run_episode()

            transitions.append(ep_transitions)
            if self.persist and (episode_count + 1) % min(self.persist_interval, self.total_episodes) == 0:
                self.persist_transitions(self.persist_dir, transitions)
                transitions = []
            episode_count += 1

    def run_episode(self):
        s = self.environment.start(self.environment.sample_state())
        a = self.agent.start(s)
        done = False
        step_count = 0
        transitions = []
        while not (done or step_count == self.max_steps_per_ep):
            ns = self.environment.step(a)
            step_count += 1
            self.agent.update(s, a, ns, done)
            transitions.append((s, a, ns))
            a = self.agent.get_action(ns)
            s = ns

        return transitions

    def set_seed(self, seed):
        if seed != -1:
            np.random.seed(seed)

    def persist_transitions(self, persist_dir, transitions):
        for ep_transitions in transitions:
            for transition in ep_transitions:
                state, action, next_state = transition
                delta = next_state - state
                fname = str(self.transition_count)
                with open(os.path.join(persist_dir, fname), "wb") as f:
                    pickle.dump((state, action, delta), f)
                self.transition_count += 1
