"""
Goal:
-   For each run:
    -   Load ensemble checkpoints at regular intervals
    -   Make epistemic/aleatoric estimates at regular intervals in the domain of the 1d function
    -   Save one figure
"""
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch

import core.model.network
import core.agent_env.environment
from core.model.ensemble import Ensemble
from core.config import ConfigLoader
from core.utils import ensure_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--num-runs', default=1, type=int)
    parser.add_argument('--config-file', default='config_files/1d_random_spawn_10k_v0/gauss_network_1d_ensmbl_test.json', help='Configuration File')
    parser.add_argument('--num-samples', default=5000)

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_loader = ConfigLoader(os.path.join(project_root, args.config_file))
    for run in range(args.num_runs):
        cfg = cfg_loader.parse(run)
        out_dir = os.path.join("analysis/plots", cfg.dataset, os.path.splitext(os.path.basename(args.config_file))[0])
        ensure_dirs([out_dir])
        out_path = os.path.join(out_dir, "{run}.png")

        env = getattr(core.agent_env.environment, cfg.env)()
        kwargs = dict(cfg.__dict__)
        Network = getattr(core.model.network, cfg.network)

        networks = []
        for k in range(cfg.num_networks):
            n = Network(**kwargs).to(device) if torch.cuda.is_available() else Network(**kwargs)
            networks.append(n)
        network = Ensemble(networks)

        print('==> Evaluation: config_file: {}'.format(args.config_file))
        mean_x, mean_y = env.mean_function()
        samples_x, samples_y = env.generate_samples()
        epoch_range = range(0, cfg.total_epochs, cfg.save_frequency)
        sns.set(style="darkgrid")
        fig, axs = plt.subplots(nrows=len(epoch_range), ncols=2, figsize=(6*2, 6*len(epoch_range)))


        axs[0][0].set_title("Aleatoric Uncertainty")
        axs[0][1].set_title("Epistemic Uncertainty")
        for k, epoch in enumerate(epoch_range):

            network.resume_checkpoint(cfg.get_interval_resume_path().format(epoch=epoch))

            low, high = env.domain()
            range_x = np.linspace(low-1.0, high+1.0, num=300)
            mean = np.zeros_like(range_x)
            aleatoric = np.zeros_like(range_x)
            epistemic = np.zeros_like(range_x)
            for idx, x in enumerate(range_x):
                x = torch.FloatTensor([x]).unsqueeze(0)
                mean[idx], epistemic[idx], aleatoric[idx] = network.predictive_mean_epistemic_aleatoric(x, 0)

            # Doing some drawing
            axs[k][0].plot(mean_x, mean_y, sns.xkcd_rgb["black"])
            axs[k][0].scatter(samples_x, samples_y, c=sns.xkcd_rgb["light blue"], marker='.')

            axs[k][1].plot(mean_x, mean_y, sns.xkcd_rgb["black"])
            axs[k][1].scatter(samples_x, samples_y, c=sns.xkcd_rgb["light blue"], marker='.')

            axs[k][0].plot(range_x, mean, c=sns.xkcd_rgb['dark purple'])
            axs[k][0].fill_between(range_x, mean - aleatoric, mean + aleatoric,
                                alpha=0.3, facecolor=sns.xkcd_rgb['light red'])
            axs[k][0].fill_between(range_x, mean - 2 * aleatoric, mean + 2 * aleatoric,
                                alpha=0.1, facecolor=sns.xkcd_rgb['light red'])
            axs[k][0].set_ylim((-6, 6))

            axs[k][1].plot(range_x, mean, c=sns.xkcd_rgb['dark purple'])
            axs[k][1].fill_between(range_x, mean - epistemic, mean + epistemic,
                                alpha=0.3, facecolor=sns.xkcd_rgb['light red'])
            axs[k][1].fill_between(range_x, mean - 2 * epistemic, mean + 2 * epistemic,
                                alpha=0.1, facecolor=sns.xkcd_rgb['light red'])
            axs[k][1].set_ylim((-6, 6))
            axs[k][0].set_ylabel("Epoch: {}".format(epoch))

        plt.savefig(out_path.format(run=run), bbox_inches='tight')

