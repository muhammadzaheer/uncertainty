import argparse

from core.agent_env.agent import Agent
from core.agent_env.environment import NoiseWorld
from core.agent_env.experiment import RandomSpawnExperiment
from core.config import DataConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate_data')
    parser.add_argument('--dataset', default='test_dataset', help='Name of the dataset')
    parser.add_argument('--num-train-samples', default=5000, type=int)
    parser.add_argument('--exp', default='RandomSpawn')

    args = parser.parse_args()
    print('==> Dataset: {}  |  Tranining Samples {}  | Exp: {}'.format(args.dataset, args.num_train_samples, args.exp))

    if args.exp == 'RandomSpawn':
        config = DataConfig(args.dataset)
        agent = Agent(num_actions=4)
        env = NoiseWorld()

        # Generating Training data
        train_samples = args.num_train_samples
        exp = RandomSpawnExperiment(agent, env, total_episodes=args.num_train_samples,
                                    max_steps_per_ep=1, persist=True,
                                    persist_dir=config.get_train_dir(), seed=0)
        exp.run()

        # Generating Testing data
        test_samples = args.num_train_samples * 0.1
        exp = RandomSpawnExperiment(agent, env, total_episodes=test_samples,
                                    max_steps_per_ep=1, persist=True,
                                    persist_dir=config.get_test_dir(), seed=0)
        exp.run()