import os
import sys
import json
from datetime import datetime

import core.utils as utils


class NetworkConfig(object):
    def __init__(self, exp_name="test", run=0, param_setting=0, lr=0.0001):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.exp_name = exp_name
        self.run = run
        self.param_setting = param_setting
        self.ensemble = False

        self.batch_size = 32
        self.num_actions = 4
        self.state_dim = 2
        self.momentum = 0.9
        self.total_epochs = 40
        self.save_frequency = 5

        # Params for stochastic evaluation
        self.test_every_batches = 100
        self.batches_per_test = 100

        self.is_resume = False

        # These parameters would come in handy for evaluation
        self.sample_loss = True
        self.exp_loss = True
        self.aleatoric = False
        self.epistemic = False

    def get_tf_log_dir(self):
        root = os.path.join(self.project_root, 'data', 'output', 'tf_logs')
        root = os.path.join(root, self.exp_name, "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting))
        root = os.path.join(root, datetime.now().strftime('%D-%T'))
        utils.ensure_dirs([root])
        return root

    def get_ckpt_dir(self):
        root = os.path.join(self.project_root, 'data', 'output', 'checkpoint')
        root = os.path.join(root, self.exp_name, "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting))
        utils.ensure_dirs([root])
        ckpt_path = os.path.join(root, 'ckpt.tar')
        best_ckpt_path = os.path.join(root, 'best_ckpt.tar')
        save_path = os.path.join(root, '{epoch}_epoch_ckpt.tar')

        return ckpt_path, best_ckpt_path, save_path

    def get_log_dir(self):
        root = os.path.join(self.project_root, 'data', 'output', 'logs')
        root = os.path.join(root, self.exp_name, "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting))
        utils.ensure_dirs([root])
        return root

    def get_log_dir_format(self):
        root = os.path.join(self.project_root, 'data', 'output', 'logs')
        root = os.path.join(root, self.exp_name, "{}_run",
                            "{}_param_setting".format(self.param_setting))
        return root

    def get_resume_path(self):
        _, best_ckpt_path, _ = self.get_ckpt_dir()
        return best_ckpt_path

        # ckpt_path, best_ckpt_path, _ = self.get_ckpt_dir()
        # return ckpt_path

    def persist_config(self):
        attrs = dict(self.__dict__)
        with open(os.path.join(self.get_log_dir(), 'config.json'), 'w') as f:
            json.dump(attrs, f, indent=4, sort_keys=True)

    def log_config(self, logger):
        attrs = dict(self.__dict__)
        for param, value in sorted(attrs.items(), key=lambda x: x[0]):
            logger.info('{}: {}'.format(param, value))


class DataConfig(object):
    def __init__(self, dataset_name="test_dataset"):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.dataset_name = dataset_name

    def get_train_dir(self):
        # Setting up data, log & checkpoint paths
        root = os.path.join(self.project_root, 'data', 'input')
        root = os.path.join(root, self.dataset_name, 'train')
        utils.ensure_dirs([root])
        return root

    def get_test_dir(self):
        # Setting up data, log & checkpoint paths
        root = os.path.join(self.project_root, 'data', 'input')
        root = os.path.join(root, self.dataset_name, 'test')
        utils.ensure_dirs([root])
        return root


class EnsembleConfig(NetworkConfig):
    def __init__(self, exp_name="test", run=0, param_setting=0, lr=0.0001):
        super().__init__(exp_name, run, param_setting, lr)
        self.num_networks = 10
        self.ensemble = True
        self.epistemic = True

    def get_ckpt_dir(self):
        root = os.path.join(self.project_root, 'data', 'output', 'checkpoint')
        root = os.path.join(root, self.exp_name, "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting))
        utils.ensure_dirs([root])
        ckpt_path = os.path.join(root, '{ensmbl}_ckpt.tar')
        best_ckpt_path = os.path.join(root, 'best_{ensmbl}_ckpt.tar')
        save_path = os.path.join(root, '{epoch}_epoch_{ensmbl}_ensemble_ckpt.tar')

        return ckpt_path, best_ckpt_path, save_path


class ConfigLoader(object):
    """
    The purpose of this class is to take an index, identify a configuration
    of hyper-parameters and create a Config object
    """
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_dict = json.load(f)
        self.total_combinations = 1
        self.set_total_combinations()

    def set_total_combinations(self):
        if 'sweep_parameters' in self.config_dict:
            sweep_params = self.config_dict['sweep_parameters']
            # calculating total_combinations
            tc = 1
            for params, values in sweep_params.items():
                tc = tc * len(values)
            self.total_combinations = tc

    def parse(self, idx):
        config_class = getattr(sys.modules[__name__], self.config_dict['config_class'])
        cfg = config_class()

        # Populating fixed parameters
        fixed_params = self.config_dict['fixed_parameters']
        for param, value in fixed_params.items():
            setattr(cfg, param, value)

        cumulative = 1

        # Populating sweep parameters
        if 'sweep_parameters' in self.config_dict:
            sweep_params = self.config_dict['sweep_parameters']
            for param, values in sweep_params.items():
                num_values = len(values)
                setattr(cfg, param, values[int(idx/cumulative) % num_values])
                cumulative *= num_values
        cfg.run = int(idx/cumulative)
        cfg.param_setting = idx % cumulative
        self.total_combinations = cumulative
        return cfg

    def param_setting_from_id(self, idx):
        sweep_params = self.config_dict['sweep_parameters']
        param_setting = {}
        cumulative = 1
        for param, values in sweep_params.items():
            num_values = len(values)
            param_setting[param] = values[int(idx/cumulative) % num_values]
            cumulative *= num_values
        return param_setting


if __name__ == '__main__':
    cfg = NetworkConfig()
    cfg.persist_config()
