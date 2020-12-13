import pickle
import json

import pandas as pd
import matplotlib.pyplot as plt
from torch import load, from_numpy, no_grad
import numpy as np

from utils.reactions import reaction_fps

from lightgbm import LGBMClassifier


def get_arguments_for_classifiers(val_ds_path, model_path, model_config_path):
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
    labels = ds["class"].values
    points = ds.rxn.apply(lambda x: reaction_fps(x,
                                                 fp_method=fp_method,
                                                 **params)
                          ).values
    points = np.stack(points)
    points = from_numpy(points).float()
    with no_grad():
        embs = model(points)
    embs = embs.detach().numpy()
    return points, embs, labels


def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


def fit_lgbm(train_orig, train_embs, train_labels):
    clf = LGBMClassifier()
    clf.fit(train_embs, train_labels)
    xx, yy = get_grid(train_embs)
    predicted = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, predicted)
    plt.figure(figsize=(20, 18))
    plt.scatter(train_embs[:, 0], train_embs[:, 1],
                c=train_labels, s=2, alpha=0.9,
                edgecolors='black', linewidth=1.5)
    plt.show()


if __name__ == '__main__':
    val_path = "../data/validation_b.csv"
    flnm = "morg30_agents_epoch_10"
    trained_model_path = f"../model/{flnm}.pt"
    trained_model_config_path = f"../model/{flnm}.json"
    X, X_emb, Y = get_arguments_for_classifiers(val_path, trained_model_path, trained_model_config_path)
    fit_lgbm(X, X_emb, Y)
