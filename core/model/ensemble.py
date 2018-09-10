import numpy as np
import torch
from torch.autograd import Variable


class Ensemble(object):
    def __init__(self, networks):
        self.networks = networks

    def init_weights(self):
        for n in self.networks:
            n.init_weight()

    def forward(self, x, a):
        ensmbl_y = []
        for n in self.networks:
            ensmbl_y.append(n.forward(x, a))

        return ensmbl_y

    def fit(self, x, a, y):
        ensmbl_loss = []
        for n in self.networks:
            ensmbl_loss.append(n.fit(x, a, y))
        return np.mean(ensmbl_loss)

    def evaluate(self, x, a, y):
        ensmbl_loss = []
        for n in self.networks:
            ensmbl_loss.append(n.evaluate(x, a, y))
        return np.mean(ensmbl_loss)

    def predict_mean(self, x, a):
        mean, _ = self.predictive_mean_variance(x, a)
        return mean

    def predict_aleatoric_variance(self, x, a):
        try:
            predictions = []
            for n in self.networks:
                predictions.append(n.predict_aleatoric_variance(x, a))
            return np.mean(predictions)
        except NotImplementedError:
            return 0.0

    def predict_epistemic_variance(self, x, a):
        # TODO: Is max the right strategy to get an estimate?
        mean, variance = self.predictive_mean_variance(x, a)
        return np.asscalar(torch.max(variance).detach().numpy())

    def predictive_mean_variance(self, x, a):
        predictions = []
        for n in self.networks:
            n.eval()
            # x = Variable(x)
            # predictions.append(n.forward(x, a))
            predictions.append(n.predict_mean(x, a))
        predictions = torch.cat(predictions)
        mean = torch.mean(predictions, dim=0)
        var = torch.var(predictions, dim=0)
        return mean, var

    def train(self):
        for n in self.networks:
            n.train()

    def eval(self):
        for n in self.networks:
            n.eval()

    def save_checkpoint(self, loss, epoch, ckpt_path):
        for n_idx in range(len(self.networks)):
            self.networks[n_idx].save_checkpoint(loss, epoch, ckpt_path.format(epoch=epoch, ensmbl=n_idx))

    def copy_checkpoint(self, from_path, to_path, epoch):
        # This abstraction is important because checkpoint can be split into
        # multiple files e.g. ensembles
        for n_idx in range(len(self.networks)):
            self.networks[n_idx].copy_checkpoint(from_path.format(ensmbl=n_idx),
                                                 to_path.format(ensmbl=n_idx), epoch)

    def resume_checkpoint(self, resume_path, map_location=None):
        for n_idx in range(len(self.networks)):
            path = resume_path.format(ensmbl=n_idx)
            start_epoch = self.networks[n_idx].resume_checkpoint(path, map_location)
        return start_epoch

    def encode_action(self, a):
        return self.networks[0].encode_action(a)

    def encode_state(self, s):
        return self.networks[0].encode_state(s)

    def write_summary(self, summary_writer):
        pass


if __name__ == '__main__':
    from core.model.network import ExpectationNetwork
    networks = []
    for k in range(10):
        networks.append(ExpectationNetwork())
    ens = Ensemble(networks=networks)
    ens_predictions = ens.forward(Variable(torch.rand(5, 1)))
    ens_losses = ens.fit(torch.rand(5, 1), torch.rand(5, 1))
    print(ens.predictive_mean_variance(torch.rand(1, 1)))