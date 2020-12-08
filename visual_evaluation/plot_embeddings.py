import pickle
import json

import pandas as pd
import matplotlib.pyplot as plt
from torch import load, from_numpy, no_grad
from torch.nn import Module
from torch.utils.data import Dataset
from numpy import stack as np_stack

from model import get_batch_embeddings
from config import config
from utils.reactions import reaction_fps


def plot_embs(trained_model: Module,
              untrained_ref_model: Module,
              ds: Dataset,
              bs=config.training_params["batch_size"]):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for ref_embs_batch, labels_batch in get_batch_embeddings(untrained_ref_model, ds, bs):
        x_init = ref_embs_batch[:, 0]
        y_init = ref_embs_batch[:, 1]
        ax[0].scatter(x_init.cpu(), y_init.cpu(), c=labels_batch, s=2, cmap="hsv")
    for trained_embs_batch, labels_batch in get_batch_embeddings(trained_model, ds, bs):
        x = trained_embs_batch[:, 0]
        y = trained_embs_batch[:, 1]
        ax[1].scatter(x.cpu(), y.cpu(), c=labels_batch, s=2, cmap="hsv")
    ax[0].set_title("Before training")
    ax[1].set_title("After training")
    plt.suptitle("Final embedding space")
    plt.show()


def plot_rxn_holdout_validation(val_ds_path, model_path, model_config_path):
    with open("../data/visual_validation/rxnClasses.pickle", "rb") as f:
        classes = pickle.load(f)
    with open(model_config_path) as conf_f:
        mod_conf = json.load(conf_f)
        settings = mod_conf["settings"]
    fp_method = settings["fp_method"]
    params = {"n_bits": settings["n_bits"],
              "fp_type": settings["fp_type"],
              "include_agents": settings["include_agents"],
              "agent_weight": settings["agent_weight"],
              "non_agent_weight": settings["non_agent_weight"],
              "bit_ratio_agents": settings["bit_ratio_agents"]
              }
    dev = "cpu"
    model = load(model_path, map_location=dev)
    model.eval()
    ds = pd.read_csv(val_ds_path, sep=';', header=None)
    ds.columns = ["rxn", "class"]
    plt.figure(figsize=(15, 12))
    for i in range(1, 10 + 1):
        subset = ds[ds["class"] == i]
        rxn_label = classes[str(i)]
        points = subset.rxn.apply(lambda x: reaction_fps(x,
                                                         fp_method=fp_method,
                                                         **params)
                                  ).values
        points = np_stack(points)
        points = from_numpy(points).float()
        with no_grad():
            embs = model(points)
        embs = embs.detach().numpy()
        plt.scatter(embs[:, 0], embs[:, 1], label=rxn_label, s=5)
    plt.legend(markerscale=3)
    plt.title(model_path.split("/")[-1])
    plt.show()


if __name__ == '__main__':
    val_path = "../data/visual_validation/validation_b.csv"
    trained_model_path = "../model/osvklg_epoch_5.pt"
    trained_model_config_path = "../model/osvklg_epoch_5.json"
    plot_rxn_holdout_validation(val_path, trained_model_path, trained_model_config_path)
