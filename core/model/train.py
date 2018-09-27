import time
import logging
import numpy as np
from torch.utils.data import DataLoader

from core.model.data_loader import Continuous2DWorldTransitions, ToTensor
from core.utils import move_to_gpu


class Trainer(object):
    def __init__(self, cfg, train_dir, test_dir, batch_size, network, summary_writer, total_epochs,
                 eval_frequency=100, num_eval_batches=20, save_frequency=5,
                 epoch_log=None, step_log=None):
        self.cfg = cfg
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.network = network
        self.summary_writer = summary_writer
        self.total_epochs = total_epochs
        self.save_frequency = save_frequency

        # eval frequency in number of updates to the network (i.e. number of batches)
        self.eval_frequency = eval_frequency
        self.num_eval_batches = num_eval_batches

        self.train_loader, self.test_loader = self.get_data_loaders()

        # Vars for tracking the training
        self.curr_epoch = 0
        self.curr_batch = 0
        self.best_loss = np.inf
        self.start_time = None

        self.epoch_logger = logging.getLogger(epoch_log) if epoch_log is not None else None
        self.step_logger = logging.getLogger(step_log) if step_log is not None else None

    def get_data_loaders(self):
        train_set = Continuous2DWorldTransitions(self.train_dir, transform=ToTensor())
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        test_set = Continuous2DWorldTransitions(self.test_dir, transform=ToTensor())
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def train_epoch(self):
        losses = []
        for i, batch in enumerate(self.train_loader):
            state, action, delta = batch['state'], batch['action'], batch['delta']
            state, action, delta = move_to_gpu([state, action, delta])
            action = self.network.encode_action(action)
            self.network.train()
            loss = self.network.fit(state, action, delta)
            self.summary_writer.add_scalar('loss/step/train_loss', loss, self.curr_epoch*len(self.train_loader) + i)
            self.network.write_summary(self.summary_writer)
            losses.append(loss)

            # if self.curr_batch % self.eval_frequency == 0:
            #     test_loss = self.evaluate_batches()
            #     self.summary_writer.add_scalar('loss/step/test_loss', test_loss, self.curr_batch)
            #     self.step_logger.info("Time {:12s} | Epoch: {} | Step: {} | Loss: {}".format(
            #         time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time)),
            #         self.curr_epoch, self.curr_batch, test_loss))
            self.curr_batch += 1
        return np.mean(losses)

    def train(self, is_resume=False):
        self.start_time = time.time()
        self.curr_epoch = self.network.resume_checkpoint(self.cfg.get_resume_path()) if is_resume else 0
        self.best_loss = np.inf
        for self.curr_epoch in range(self.curr_epoch, self.total_epochs):
            if (self.curr_epoch % self.save_frequency == 0) or (self.curr_epoch == self.total_epochs - 1):
                train_loss = self.evaluate()
                self.summary_writer.add_scalar('loss/epoch/train_loss', train_loss, self.curr_epoch)
            self.train_epoch()

    def evaluate_batches(self):
        # Evaluates a fixed number of random batches from the test-set
        self.network.eval()
        test_iter = iter(self.test_loader)
        num_batches = min(self.num_eval_batches, len(self.test_loader))
        loss = 0.0
        for _ in range(num_batches):
            batch = test_iter.next()
            state, action, delta = batch['state'], batch['action'], batch['delta']
            state, action, delta = move_to_gpu([state, action, delta])
            action = self.network.encode_action(action)
            loss += self.network.evaluate(state, action, delta)
        loss /= num_batches
        return loss

    def evaluate_epoch(self):
        # Evaluates the entire test-set
        self.network.eval()
        loss = 0.0
        for i, batch in enumerate(self.test_loader):
            state, action, next_state = batch['state'], batch['action'], batch['delta']
            state, action, next_state = move_to_gpu([state, action, next_state])
            action = self.network.encode_action(action)
            loss += self.network.evaluate(state, action, next_state)
        loss /= len(self.test_loader)
        return loss

    def evaluate(self):
        test_loss = self.evaluate_epoch()
        self.summary_writer.add_scalar('loss/epoch/test_loss', test_loss, self.curr_epoch)
        # Overwriting the latest checkpoint
        ckpt_path, best_ckpt_path, save_path = self.cfg.get_ckpt_dir()
        self.network.save_checkpoint(test_loss, self.curr_epoch, ckpt_path)
        # Copying to the epoch checkpoint
        self.network.copy_checkpoint(ckpt_path, save_path, self.curr_epoch)

        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.network.copy_checkpoint(ckpt_path, best_ckpt_path, self.curr_epoch)

        self.epoch_logger.info("Time {:12s} | Epoch: {} | Loss: {}".format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time)),
            self.curr_epoch, test_loss))

        return self.best_loss

