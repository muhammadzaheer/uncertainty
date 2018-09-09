import os
import argparse

import torch
from tensorboardX import SummaryWriter

import core.model.network
from core.model.ensemble import Ensemble
from core.model.train import Trainer
from core.config import DataConfig, ConfigLoader
from core.utils import setup_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--id', default=0)
    parser.add_argument('--config-file', default='config_files/test.json', help='Configuration File')
    args = parser.parse_args()

    cfg = ConfigLoader(args.config_file).parse(args.id)
    cfg.persist_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Network = getattr(core.model.network, cfg.network)
    if cfg.ensemble:
        networks = []
        for k in range(cfg.num_networks):
            n = Network(cfg.lr).to(device) if torch.cuda.is_available() else Network(cfg.lr)
            networks.append(n)
        network = Ensemble(networks)
    else:
        network = Network(cfg.lr).to(device) if torch.cuda.is_available() else Network(cfg.lr)
    data_cfg = DataConfig(cfg.dataset)

    # Setting up train/test dirs
    train_dir = data_cfg.get_train_dir()
    test_dir = data_cfg.get_test_dir()

    # Setting up summary-writer for tensorboard logs
    sw = SummaryWriter(cfg.get_tf_log_dir())

    # Setting up loggers
    log_dir = cfg.get_log_dir()
    epoch_log = os.path.join(log_dir, 'epoch_log')
    epoch_logger = setup_logger(epoch_log, stdout=True)
    cfg.log_config(epoch_logger)
    step_log = os.path.join(log_dir, 'step_log')
    step_logger = setup_logger(step_log, stdout=False)
    cfg.log_config(step_logger)

    # Creating the trainer object
    trainer = Trainer(cfg, train_dir, test_dir, cfg.batch_size,
                      network, sw, cfg.total_epochs,
                      epoch_log=epoch_log, step_log=step_log)
    trainer.train(is_resume=cfg.is_resume)
