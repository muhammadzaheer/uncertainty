import os
import pickle

from core.config import ConfigLoader
from analysis.visualizer import HeatMaps


def parse(log_path):
    with open(log_path, "rb") as f:
        map = pickle.load(f)
    return map

def draw_expectation():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/test.json"), 0)]
    path_formatters = []
    for cf, param_setting in config_files:
        cfg = ConfigLoader(cf).parse(param_setting)
        logdir_format = cfg.get_log_dir_format()
        path_format = os.path.join(logdir_format, "sample_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "exp_loss")
        path_formatters.append(path_format)

    num_runs = 1
    cols = ["sample_loss", "exp_loss"]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/test_map.png", cols=cols)
    v.draw()


def draw_ensemble():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/test_ensmbl.json"), 0)]
    path_formatters = []
    for cf, param_setting in config_files:
        cfg = ConfigLoader(cf).parse(param_setting)
        logdir_format = cfg.get_log_dir_format()
        path_format = os.path.join(logdir_format, "sample_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "exp_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "epistemic")
        path_formatters.append(path_format)

    num_runs = 1
    cols = ["sample_loss", "exp_loss", "epistemic"]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/test_map_ensemble.png", cols=cols)
    v.draw()


def draw_ensemble_aleatoric():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/test_ensmbl_aleatoric.json"), 0)]
    path_formatters = []
    for cf, param_setting in config_files:
        cfg = ConfigLoader(cf).parse(param_setting)
        logdir_format = cfg.get_log_dir_format()
        path_format = os.path.join(logdir_format, "sample_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "exp_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "aleatoric")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "epistemic")
        path_formatters.append(path_format)


    num_runs = 1
    cols = ["sample_loss", "exp_loss", "aleatoric", "epistemic"]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/test_map_ensemble_aleatoric.png", cols=cols)
    v.draw()

if __name__ == '__main__':
    # draw_expectation()
    # draw_ensemble()
    draw_ensemble_aleatoric()




