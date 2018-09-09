import os
import shutil
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self, lr=0.001, state_dim=2, num_actions=4):
        super(Network, self).__init__()
        self.lr = lr
        self.state_dim = state_dim
        self.num_actions = num_actions
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


class ExpectationNetwork(Network):
    def __init__(self, lr=0.001, state_dim=2, num_actions=4):
        super(ExpectationNetwork, self).__init__(lr, state_dim, num_actions)

        self.fc1 = nn.Linear(in_features=state_dim, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

        self.w_enc = nn.Linear(in_features=10, out_features=10)
        self.w_a = nn.Linear(in_features=num_actions, out_features=10)
        self.w_dec = nn.Linear(in_features=10, out_features=10)

        self.fc3 = nn.Linear(in_features=10, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=state_dim)

        self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
    def __init__(self, lr=0.001, state_dim=2, num_actions=4):
        super(GaussianVarianceNetwork, self).__init__(lr, state_dim, num_actions)

        self.fc1 = nn.Linear(in_features=state_dim, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

        self.w_enc = nn.Linear(in_features=10, out_features=10)
        self.w_a = nn.Linear(in_features=num_actions, out_features=10)
        self.w_dec = nn.Linear(in_features=10, out_features=10)

        self.fc3 = nn.Linear(in_features=10, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=state_dim*2)

        self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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


if __name__ == "__main__":
    net = ExpectationNetwork(lr=0.001)
    action = torch.zeros(1, 4)
    action[0][1] = 1

    print(net.forward(Variable(torch.rand(1, 2)),
                      Variable(action)))

    print(net.fit(torch.rand(1, 2), action,
                  torch.rand(1, 2)))
