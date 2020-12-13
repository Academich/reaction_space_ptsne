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
    return embs, labels


def fit_lgbm_on_embs(train_embs, train_labels):
    clf = LGBMClassifier(n_jobs=4, random_state=50)
    clf.fit(train_embs, train_labels)
    predicted = clf.predict(train_embs)
    return np.mean(predicted == train_labels)
    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))
    #
    # for i in range(1, 10 + 1):
    #     subset_true = train_embs[(train_labels == i).nonzero()]
    #     subset_pred = train_embs[(predicted == i).nonzero()]
    #     rxn_label = i
    #     ax[0].scatter(subset_pred[:, 0], subset_pred[:, 1], label=rxn_label, s=5)
    #     ax[1].scatter(subset_true[:, 0], subset_true[:, 1], label=rxn_label, s=5)
    # plt.legend(markerscale=3)
    # plt.show()


if __name__ == '__main__':
    val_path = "data/visual_validation/validation_b.csv"
    names = ["morg10_epoch_10", "morg30_epoch_10", "morg100_epoch_10", "morg500_epoch_10",
             "pair10_epoch_10", "pair30_epoch_10", "pair100_epoch_10", "pair500_epoch_10",
             "toto30_epoch_10",
             "morg30_epoch_40"]
    for flnm in names:
        trained_model_path = f"saved_models/{flnm}.pt"
        trained_model_config_path = f"saved_models/{flnm}.json"
        X_emb, Y = get_arguments_for_classifiers(val_path, trained_model_path, trained_model_config_path)
        score = fit_lgbm_on_embs(X_emb, Y)
        print(f"{flnm}: {score}")

        # morg10_epoch_10: 0.83668
        # morg30_epoch_10: 0.83638
        # morg100_epoch_10: 0.83596
        # morg500_epoch_10: 0.81924
        # pair10_epoch_10: 0.73018
        # pair30_epoch_10: 0.75314
        # pair100_epoch_10: 0.68988
        # pair500_epoch_10: 0.7264
        # toto30_epoch_10: 0.75048
        # morg30_epoch_40: 0.85916
