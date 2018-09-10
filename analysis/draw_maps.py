import os
import pickle

from core.config import ConfigLoader
from analysis.visualizer import HeatMaps


def parse(log_path):
    with open(log_path, "rb") as f:
        map = pickle.load(f)
    return map


def draw_exp():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/exp_network.json"), 0)]
    path_formatters = []
    for cf, param_setting in config_files:
        cfg = ConfigLoader(cf).parse(param_setting)
        logdir_format = cfg.get_log_dir_format()
        path_format = os.path.join(logdir_format, "sample_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "exp_loss")
        path_formatters.append(path_format)

    num_runs = 10
    cols = ["sample_loss", "exp_loss"]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/exp_network.png", cols=cols, vrange=vrange)
    v.draw()


def draw_exp_ensmbl():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/exp_network_ensmbl.json"), 0)]
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

    num_runs = 10
    cols = ["sample_loss", "exp_loss", "epistemic"]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/exp_network_ensmbl.png", cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_ensmbl():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/gauss_network_ensmbl.json"), 0)]
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

    num_runs = 10
    cols = ["sample_loss", "exp_loss", "aleatoric", "epistemic"]
    vrange = [None, None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/gauss_network_ensmbl.png", cols=cols, vrange=vrange)
    v.draw()


def draw_gauss():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/gauss_network.json"), 0)]
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

    num_runs = 10
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/gauss_network.png", cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v2():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, "config_files/gauss_network_v2.json"), 1)]
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

    num_runs = 1
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path="plots/gauss_network_v2.png", cols=cols, vrange=vrange)
    v.draw()


if __name__ == '__main__':
    draw_exp()
    draw_exp_ensmbl()
    draw_gauss()
    draw_gauss_ensmbl()

    # draw_gauss_v2()




