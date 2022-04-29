import yaml
import os

config_name = "config.yaml"

def get_conf_dict(config_name):
    if os.path.exists(config_name):
        conf_path = config_name
    else:
        conf_path = f"../{config_name}"
    with open(conf_path) as f:
        conf_dict = yaml.safe_load(f)
    return conf_dict

class Config:

    def __init__(self, conf_dict):
        self.dev = conf_dict["device"]
        self.seed = conf_dict["seed"]
        self.save_dir_path = conf_dict["save_dir_path"]
        self.epochs_to_save_after = conf_dict["epochs_to_save_after"]

        self.problem_settings = conf_dict["problem_settings"]
        self.optimization_conf = conf_dict["optimization"]
        self.training_params = conf_dict["training"]


