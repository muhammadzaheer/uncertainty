import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter

from data_loader import Continuous2DWorldTransitions, ToTensor
from utils import move_to_gpu, boolean_string
from ensemble import Ensemble


def train_epoch(network, data_loader, summary_writer, epoch):
    network.train()
    losses = []
    for i, batch in enumerate(data_loader):
        state, action_int, delta = batch['state'], batch['action'], \
                                        batch['delta']
        state, action_int, delta = move_to_gpu([state, action_int, delta])
        action = cfg.FloatTensor(action_int.size()[0], cfg.num_actions)
        action.zero_()
        action.scatter_(dim=1, index=action_int,
                        src=torch.ones(action_int.size()).type(cfg.FloatTensor))

        l = network.fit(state, action, delta)
        summary_writer.add_scalar('loss/training_loss', l, epoch*len(train_loader) + i)
        losses.append(l)

        if (i+1) % cfg.test_every_batches == 0:
            training_step = len(train_loader) * epoch + i
            test_loss = eval_batches(net, test_loader, cfg.batches_per_test)
            print('Training Step: {} | Sample test Loss: {:.10f}'.format(
                            training_step, test_loss
                        ))
            summary_writer.add_scalar('loss/sample_test_loss', test_loss, training_step)
            print("Epoch: {} | Sample_test_loss: {}".format(epoch, test_loss))
    return losses


def eval_batches(network, data_loader, num_batches):
    # Evaluates a fixed number of random batches from the test-set
    network.eval()
    loss = 0.0
    test_iter = iter(data_loader)
    for x in range(min(num_batches, len(test_loader))):
        batch = test_iter.next()
        state, action_int, delta = batch['state'], batch['action'], \
                                   batch['delta']
        state, action_int, delta = move_to_gpu([state, action_int, delta])
        action = cfg.FloatTensor(action_int.size()[0], cfg.num_actions)
        action.zero_()
        action.scatter_(dim=1, index=action_int,
                        src=torch.ones(action_int.size()).type(cfg.FloatTensor))

        loss += network.evaluate(state, action, delta)
    loss /= min(num_batches, len(data_loader))
    return loss


def evaluate_epoch(network, data_loader):
    # Evaluates the entire test-set
    network.eval()
    loss = 0.0
    for i, batch in enumerate(data_loader):
        state, action_int, next_state = batch['state'], batch['action'], \
                                        batch['delta']
        state, action_int, next_state = move_to_gpu([state, action_int, next_state])
        action = cfg.FloatTensor(action_int.size()[0], cfg.num_actions)
        action.zero_()
        action.scatter_(dim=1, index=action_int,
                        src=torch.ones(action_int.size()).type(cfg.FloatTensor))

        loss += network.evaluate(state, action, next_state)
    loss /= len(data_loader)
    return loss


def evaluate(config, network, data_loader, summary_writer, epoch, best_loss):
    test_loss = evaluate_epoch(network, data_loader)
    summary_writer.add_scalar('loss/test_loss', test_loss, epoch)
    # Checkpointing
    is_best = False
    if test_loss < best_loss:
        is_best = True
        best_loss = test_loss

    # Overwriting the latest checkpoint
    network.save_checkpoint(test_loss, epoch, config.ckpt_path)
    # Copying to the epoch checkpoint
    # network.copy_checkpoint(config.ckpt_path, config.save_path, epoch)
    if is_best:
        # Overwriting the best checkpoint
        network.copy_checkpoint(config.ckpt_path, config.best_ckpt_path, epoch)
    print('=> Epoch {:5d}| Test Loss: {:.8f}'.format(epoch, test_loss))

    return best_loss


def get_data_loaders(config):
    train_set = Continuous2DWorldTransitions(config.train_dir, transform=ToTensor())
    trainer = DataLoader(train_set,
                         batch_size=config.batch_size,
                         shuffle=True)

    test_set = Continuous2DWorldTransitions(config.test_dir, transform=ToTensor())
    tester = DataLoader(test_set,
                        batch_size=config.batch_size,
                        shuffle=True)
    return trainer, tester


def setup_network_architecture(config):
    # Setting up the architecture
    nn_module = nn.ModuleList()
    n1, n2 = config.num_units[0], config.num_units[1]
    nn_module.append(nn.Linear(in_features=config.state_dim, out_features=n1))
    nn_module.append(nn.Linear(in_features=n1, out_features=n2))
    nn_module.append(nn.Linear(in_features=config.num_actions, out_features=n2))
    nn_module.append(nn.Linear(in_features=n2, out_features=n1))
    nn_module.append(nn.Linear(in_features=n1, out_features=config.state_dim))
    nn_module.append(nn.Linear(in_features=config.state_dim, out_features=config.output_units))
    return nn_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--dataset', default='Markov_delta_5k', help='Name of the dataset')
    parser.add_argument('--run', default="0")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--variance-method', default='mse',
                        help='square-dev: VarianceSquareDevNetwork, '
                        'exp-square: VarianceExpSquareNetwork ,'
                        'gaussian: VarianceGaussianNetwork')
    parser.add_argument('--ensemble', default=False, type=boolean_string)
    parser.add_argument('--num-nets-ensemble', default=10, type=int)

    args = parser.parse_args()

    if args.ensemble:
        from config import EnsembleConfig as Config
    else:
        from config import Config

    cfg = Config(exp_name=args.exp_name, dataset=args.dataset, units=(100, 50), lr=args.lr, run=args.run)
    train_loader, test_loader = get_data_loaders(cfg)

    if args.variance_method == 'square-dev':
        from network import VarianceSquareDevNetwork as Network
    elif args.variance_method == 'exp-square':
        from network import VarianceExpSquareNetwork as Network
    elif args.variance_method == 'gaussian':
        from network import VarianceGaussianNetwork as Network
    elif args.variance_method == 'mse':
        from network import MeanNetwork as Network
    else:
        raise NotImplementedError

    cfg.output_units = cfg.state_dim if args.variance_method == 'mse' else cfg.state_dim * 2
    if args.ensemble:
        networks = []
        for k in range(args.num_nets_ensemble):
            cfg.nn_module = setup_network_architecture(cfg)
            net = Network(cfg).cuda() if cfg.use_cuda else Network(cfg)
            networks.append(net)
        net = Ensemble(networks)
    else:
        cfg.nn_module = setup_network_architecture(cfg)
        net = Network(cfg).cuda() if cfg.use_cuda else Network(cfg)

    print("dataset: {} | run: {} | variance method: {}".format(
          args.dataset, args.run, args.variance_method))
    cfg.persist_config()

    start_epoch = net.resume_checkpoint(cfg.resume_path) if cfg.is_resume else 0
    sw = SummaryWriter(cfg.logs_path)
    best_loss = np.inf
    for curr_epoch in range(start_epoch, cfg.num_epochs + 1):
        # Checkpoint every-fixed interval
        if (curr_epoch % cfg.save_frequency == 0) or (curr_epoch == cfg.num_epochs):
            best_loss = evaluate(cfg, net, test_loader, sw, curr_epoch, best_loss)
        train_epoch(net, train_loader, sw, curr_epoch)
    sw.close()
