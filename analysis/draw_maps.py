import os
import pickle

from core.config import ConfigLoader
from analysis.visualizer import HeatMaps


def parse(log_path):
    with open(log_path, "rb") as f:
        map = pickle.load(f)
    return map


def draw_exp(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
    path_formatters = []
    for cf, param_setting in config_files:
        cfg = ConfigLoader(cf).parse(param_setting)
        logdir_format = cfg.get_log_dir_format()
        path_format = os.path.join(logdir_format, "sample_loss")
        path_formatters.append(path_format)
        path_format = os.path.join(logdir_format, "exp_loss")
        path_formatters.append(path_format)

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss"]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_exp_ensmbl(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "epistemic"]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_ensmbl(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric", "epistemic"]
    vrange = [None, None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v2a(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v2b(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v3a(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v3b(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v3c(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


def draw_gauss_v3d(config_file, out_file, num_runs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    config_files = [(os.path.join(project_root, config_file), 0)]
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

    num_runs = num_runs
    cols = ["sample_loss", "exp_loss", "aleatoric"]
    # vrange = [None, None, (0.0, 0.004)]
    vrange = [None, None, None]
    v = HeatMaps(path_formatters, num_runs, parser_func=parse,
                 save_path=out_file, cols=cols, vrange=vrange)
    v.draw()


if __name__ == '__main__':
    # draw_exp(config_file="config_files/random_spawn_20k_v2/exp_network.json", out_file="plots/random_spawn_20k_v2/exp_network.png", num_runs=3)
    # draw_exp_ensmbl()
    # draw_gauss(config_file="config_files/random_spawn_20k_v2/gauss_network_1d.json", out_file="plots/random_spawn_20k_v2/gauss_network.png", num_runs=7)
    # draw_gauss(config_file="config_files/random_spawn_10k_v1/gauss_network_1d.json", out_file="plots/random_spawn_10k_v1/gauss_network.png", num_runs=6)
    # draw_gauss(config_file="config_files/random_spawn_5k_v0/gauss_network_1d.json", out_file="plots/random_spawn_5k_v0/gauss_network.png", num_runs=6)
    # draw_gauss_ensmbl()
    # draw_gauss_v2a()
    # draw_gauss_v2b()
    draw_gauss_v3a(config_file="config_files/random_spawn_20k_v2/gauss_network_v3a.json", out_file="plots/random_spawn_20k_v2/gauss_network_v3a.png", num_runs=7)
    draw_gauss_v3b(config_file="config_files/random_spawn_20k_v2/gauss_network_v3b.json", out_file="plots/random_spawn_20k_v2/gauss_network_v3b.png", num_runs=7)

    # draw_gauss_v3a(config_file="config_files/random_spawn_10k_v1/gauss_network_v3a.json",
    #                out_file="plots/random_spawn_10k_v1/gauss_network_v3a.png", num_runs=5)
    # draw_gauss_v3b(config_file="config_files/random_spawn_10k_v1/gauss_network_v3b.json",
    #                out_file="plots/random_spawn_10k_v1/gauss_network_v3b.png", num_runs=5)
    #
    # draw_gauss_v3a(config_file="config_files/random_spawn_5k_v0/gauss_network_v3a.json",
    #                out_file="plots/random_spawn_5k_v0/gauss_network_v3a.png", num_runs=5)
    # draw_gauss_v3b(config_file="config_files/random_spawn_5k_v0/gauss_network_v3b.json",
    #                out_file="plots/random_spawn_5k_v0/gauss_network_v3b.png", num_runs=5)

    # draw_gauss_v3c()
    # draw_gauss_v3d()




