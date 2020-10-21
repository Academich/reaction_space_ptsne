import yaml

with open("config.yaml") as f:
    conf_dict = yaml.safe_load(f)


class Config:

    def __init__(self):
        self.dev = conf_dict["device"]
        self.seed = conf_dict["seed"]
        self.save_flag = bool(conf_dict['save_model'])
        self.optimization_conf = conf_dict["optimization"]
        self.training_params = conf_dict["training"]
        self.data = conf_dict["dataset"]


config = Config()
