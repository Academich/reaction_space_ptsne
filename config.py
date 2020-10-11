import yaml

with open("config.yaml") as f:
    conf_dict = yaml.safe_load(f)


class Config:

    def __init__(self):
        self.dev = conf_dict["device"]
        self.optimization_conf = conf_dict["optimization"]
        self.training_params = conf_dict["training"]


config = Config()
