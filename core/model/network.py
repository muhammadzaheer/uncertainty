import os
import shutil
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()
        self.lr = kwargs['lr']
        self.state_dim = kwargs['state_dim']
        self.num_actions = kwargs['num_actions']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight.data)
                m.bias.data.fill_(0)

        self.w_enc.weight.data.uniform_(-1.0, 1.0)
        self.w_a.weight.data.uniform_(-1.0, 1.0)
        self.w_dec.weight.data.uniform_(-1.0, 1.0)

    def forward(self, x, a):
        raise NotImplementedError

    def fit(self, x, a, y):
        raise NotImplementedError

    def evaluate(self, x, a, y):
        raise NotImplementedError

    def predict_mean(self, x, a):
        raise NotImplementedError

    def predict_aleatoric_variance(self, x, a):
        raise NotImplementedError

    def mse_loss(self, predicted_y, y):
        loss = 0.5*torch.abs(y - predicted_y)**2
        return loss.mean()

    def resume_checkpoint(self, resume_path, map_location=None):
        start_epoch = 0
        if os.path.isfile(resume_path):
            print("=> Loading checkpoint at {}".format(resume_path))
            ckpt = torch.load(resume_path, map_location=map_location)
            start_epoch = ckpt['epoch']
            self.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict((ckpt['optimizer']))
            loss = ckpt['loss']
            print('=> Loaded checkpoint at {} | epoch: {} | loss: {:.10f}'
                  .format(resume_path, start_epoch, loss))
        else:
            print('=> No checkpoint file found at {}'.format(resume_path))
        return start_epoch

    def copy_checkpoint(self, from_path, to_path, epoch):
        # This abstraction is important because checkpoint can be split into
        # multiple files e.g. ensembles
        shutil.copyfile(from_path, to_path.format(epoch=epoch))

    def save_checkpoint(self, loss, epoch, ckpt_path):
        training_state = {'loss': loss,
                          'epoch': epoch,
                          'state_dict': self.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
        torch.save(training_state, ckpt_path)

    def encode_action(self, a):
        action = torch.zeros(a.size()[0], self.num_actions, dtype=torch.float, device=self.device)
        action.zero_()
        action.scatter_(dim=1, index=a,
                        src=torch.ones(a.size(), dtype=torch.float, device=self.device))
        return action

    def encode_state(self, s):
        return torch.FloatTensor(s).unsqueeze(0)

    def write_summary(self, summary_writer):
        pass


class ExpectationNetwork(Network):
    def __init__(self, **kwargs):
        super(ExpectationNetwork, self).__init__(**kwargs)

        self.fc1 = nn.Linear(in_features=self.state_dim, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

        self.w_enc = nn.Linear(in_features=10, out_features=10)
        self.w_a = nn.Linear(in_features=self.num_actions, out_features=10)
        self.w_dec = nn.Linear(in_features=10, out_features=10)

        self.fc3 = nn.Linear(in_features=10, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=self.state_dim)

        self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        h_enc = self.w_enc(x)
        a_enc = self.w_a(a)
        h_dec = self.w_dec(torch.mul(h_enc, a_enc))

        x = F.relu(self.fc3(h_dec))
        x = self.fc4(x)

        return x

    def fit(self, x, a, y):
        x = Variable(x)
        a = Variable(a)
        y = Variable(y)

        self.optimizer.zero_grad()

        predicted_y = self.forward(x, a)
        loss = self.mse_loss(predicted_y, y)
        loss.backward()

        self.optimizer.step()

        return np.asscalar(loss.data.cpu().numpy())

    def evaluate(self, x, a, y):
        x = Variable(x, volatile=False)
        a = Variable(a, volatile=False)
        y = Variable(y, volatile=False)

        predicted_y = self.forward(x, a)
        loss = self.mse_loss(predicted_y, y)
        return np.asscalar(loss.data.cpu().numpy())

    def predict_mean(self, x, a):
        x = Variable(x, volatile=False)
        a = Variable(a, volatile=False)

        return self.forward(x, a)

    def predict_aleatoric_variance(self, x, a):
        raise NotImplementedError


class GaussianVarianceNetwork(Network):
    # V1: Same as expectation network except that we have two extra units
    def __init__(self, **kwargs):
        super(GaussianVarianceNetwork, self).__init__(**kwargs)

        self.fc1 = nn.Linear(in_features=self.state_dim, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

        self.w_enc = nn.Linear(in_features=10, out_features=10)
        self.w_a = nn.Linear(in_features=self.num_actions, out_features=10)
        self.w_dec = nn.Linear(in_features=10, out_features=10)

        self.fc3 = nn.Linear(in_features=10, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=self.state_dim*2)

        self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        h_enc = self.w_enc(x)
        a_enc = self.w_a(a)
        h_dec = self.w_dec(torch.mul(h_enc, a_enc))

        x = F.relu(self.fc3(h_dec))
        x = self.fc4(x)

        return x

    def fit(self, x, a, y):
        x = Variable(x)
        a = Variable(a)
        y = Variable(y)

        self.optimizer.zero_grad()

        predicted_y = self.forward(x, a)
        loss = self.loss(predicted_y, y)
        loss.backward()

        self.optimizer.step()

        return np.asscalar(loss.data.cpu().numpy())

    def evaluate(self, x, a, y):
        x = Variable(x, volatile=False)
        a = Variable(a, volatile=False)
        y = Variable(y, volatile=False)

        predicted_y = self.forward(x, a)
        loss = self.loss(predicted_y, y)
        return np.asscalar(loss.data.cpu().numpy())

    def predict_mean(self, x, a):
        x = Variable(x, volatile=False)
        a = Variable(a, volatile=False)

        return self.forward(x, a)[:, :self.state_dim]

    def predict_aleatoric_variance(self, x, a):
        variances = torch.exp(self.forward(x, a)[:, 2:])
        max_var = variances.max(dim=1, keepdim=True)[0]
        return np.asscalar(max_var.data.cpu().numpy())

    def loss(self, predicted_y, y):
        mu = predicted_y[:, :self.state_dim]
        logvar = predicted_y[:, self.state_dim:]
        var = torch.exp(logvar)

        squared_dev = (mu - y)**2
        determ = torch.cumprod(var, dim=1)[:, -1:]
        loss = torch.sum(squared_dev * 1./var, dim=1).unsqueeze(1) + torch.log(determ)

        return loss.mean()


class GaussianVarianceNetworkv2(Network):
    # Separate layers for exp & variane
    def __init__(self, **kwargs):
        super(GaussianVarianceNetworkv2, self).__init__(**kwargs)
        self.bound_var = kwargs['bound_var']

        self.fc1 = nn.Linear(in_features=self.state_dim, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

        self.w_enc = nn.Linear(in_features=10, out_features=10)
        self.w_a = nn.Linear(in_features=self.num_actions, out_features=10)
        self.w_dec = nn.Linear(in_features=10, out_features=10)

        self.exp_fc3 = nn.Linear(in_features=10, out_features=10)
        self.exp_fc4 = nn.Linear(in_features=10, out_features=self.state_dim)

        self.var_fc3 = nn.Linear(in_features=10, out_features=10)
        self.var_fc4 = nn.Linear(in_features=10, out_features=self.state_dim)

        self.exp_drop = nn.Dropout(p=0.2)
        self.var_drop = nn.Dropout(p=0.2)

        # Using Kurtland's Chua config out of the box
        self.max_logvar = nn.Parameter(torch.ones(2)*0.5)
        self.min_logvar = nn.Parameter(torch.ones(2)*-10)

        self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), self.lr)

        # DEBUG vars
        self.train_step = 0
        self.test_step = 0

    def forward(self, x, a):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        h_enc = self.w_enc(x)
        a_enc = self.w_a(a)
        h_dec = self.w_dec(torch.mul(h_enc, a_enc))

        exp = F.relu(self.exp_fc3(h_dec))
        exp = self.exp_drop(exp)
        exp = self.exp_fc4(exp)

        logvar = F.relu(self.var_fc3(h_dec))
        logvar = self.var_drop(logvar)
        logvar = self.var_fc4(logvar)

        if self.bound_var:
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return exp, logvar

    def fit(self, x, a, y):
        x = Variable(x)
        a = Variable(a)
        y = Variable(y)

        self.optimizer.zero_grad()

        exp_y, logvar_y = self.forward(x, a)
        loss = self.loss(exp_y, logvar_y, y)
        loss.backward()

        self.optimizer.step()

        return np.asscalar(loss.data.cpu().numpy())

    def evaluate(self, x, a, y):
        x = Variable(x, volatile=False)
        a = Variable(a, volatile=False)
        y = Variable(y, volatile=False)

        exp_y, logvar_y = self.forward(x, a)
        loss = self.loss(exp_y, logvar_y, y)
        return np.asscalar(loss.data.cpu().numpy())

    def predict_mean(self, x, a):
        x = Variable(x, volatile=False)
        a = Variable(a, volatile=False)

        exp, _ = self.forward(x, a)
        return exp

    def predict_aleatoric_variance(self, x, a):
        _, logvar = self.forward(x, a)
        var = torch.exp(logvar)
        max_var = var.max(dim=1, keepdim=True)[0]
        return np.asscalar(max_var.data.cpu().numpy())

    def loss(self, mu, logvar, y):
        var = torch.exp(logvar)

        squared_dev = (mu - y)**2
        determ = torch.cumprod(var, dim=1)[:, -1:]
        loss = torch.sum(squared_dev * 1./var, dim=1).unsqueeze(1) + torch.log(determ)

        if self.bound_var:
            loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)

        # DEBUG
        if self.training:
            self.train_step += 1
        else:
            self.test_step += 1
        return loss.mean()

    def write_summary(self, summary_writer):
        if self.training and self.bound_var:
            summary_writer.add_scalar('loss/step/maxlogvar0', self.max_logvar[0], self.train_step)
            summary_writer.add_scalar('loss/step/maxlogvar1', self.max_logvar[1], self.train_step)
            summary_writer.add_scalar('loss/step/minlogvar0', self.min_logvar[0], self.train_step)
            summary_writer.add_scalar('loss/step/minlogvar1', self.min_logvar[1], self.train_step)


if __name__ == "__main__":
    net = ExpectationNetwork(lr=0.001)
    action = torch.zeros(1, 4)
    action[0][1] = 1

    print(net.forward(Variable(torch.rand(1, 2)),
                      Variable(action)))

    print(net.fit(torch.rand(1, 2), action,
                  torch.rand(1, 2)))
