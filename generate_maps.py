import argparse

import torch

import core.model.network
import core.agent_env.environment
from core.model.ensemble import Ensemble
from core.agent_env.agent import Agent
from core.evaluation.evaluation import RandomSpawnEvaluation
from core.config import DataConfig, ConfigLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--config-file', default='config_files/test.json', help='Configuration File')
    parser.add_argument('--eval', default='RandomSpawn')
    parser.add_argument('--num-steps', default=5000)
    args = parser.parse_args()

    args = parser.parse_args()

    cfg = ConfigLoader(args.config_file).parse(args.id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Network = getattr(core.model.network, cfg.network)
    kwargs = dict(cfg.__dict__)
    if cfg.ensemble:
        networks = []
        for k in range(cfg.num_networks):
            n = Network(**kwargs).to(device) if torch.cuda.is_available() else Network(**kwargs)
            networks.append(n)
        network = Ensemble(networks)
    else:
        network = Network(**kwargs).to(device) if torch.cuda.is_available() else Network(**kwargs)

    print('==> Evaluation: {}  |  Steps {}  | config_file: {}'.format(args.eval, args.num_steps, args.config_file))
    network.resume_checkpoint(cfg.get_resume_path())

    if args.eval == 'RandomSpawn':
        config = DataConfig()
        agent = Agent(num_actions=4)
        env_class = getattr(core.agent_env.environment, cfg.env)
        env = env_class()
        # Generating Training data
        exp = RandomSpawnEvaluation(network, agent, env, total_episodes=args.num_steps,
                                    max_steps_per_ep=1, seed=0, is_exp=cfg.exp_loss,
                                    is_aleatoric=cfg.aleatoric, is_epistemic=cfg.epistemic)
        exp.run()

    log_dir = cfg.get_log_dir()
    exp.persist_maps(log_dir)