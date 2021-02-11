import json
import datetime

import pandas as pd
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


if __name__ == '__main__':
    val_path = "data/validation_b.csv"
    names = [
        "morg10_epoch_10",
        "morg30_epoch_10",
        "morg100_epoch_10",
        "morg500_epoch_10",
        # "morg10_epoch_40",
        "morg30_epoch_40",
        # "morg100_epoch_40",
        # "morg500_epoch_40",
        "pair10_epoch_40",
        "pair30_epoch_40",
        "pair100_epoch_40",
        "pair500_epoch_40",
        "toto10_epoch_40",
        "toto30_epoch_40",
        "toto100_epoch_40",
        "toto500_epoch_40"
    ]
    # names = ["morg30_struct_jacc_epoch_10"]
    lgb_scores = []
    for flnm in names:
        start = datetime.datetime.now()
        trained_model_path = f"saved_models/{flnm}.pt"
        trained_model_config_path = f"saved_models/{flnm}.json"
        X_emb, Y = get_arguments_for_classifiers(val_path, trained_model_path, trained_model_config_path)
        lgbm_score = fit_lgbm_on_embs(X_emb, Y)
        fin = datetime.datetime.now()
        lgb_scores.append((lgbm_score, flnm))
        print(f"{flnm} --> lgbm: {lgbm_score}")
    print(sorted(lgb_scores))

    # morg10_epoch_10 --> lgbm: 0.83808
    # morg30_epoch_10 --> lgbm: 0.83638
    # morg100_epoch_10 --> lgbm: 0.8335
    # morg500_epoch_10 --> lgbm: 0.8194
    # morg10_epoch_40 --> lgbm: 0.3028
    # morg30_epoch_40 --> lgbm: 0.85916
    # morg100_epoch_40 --> lgbm: 0.3028
    # morg500_epoch_40 --> lgbm: 0.3028
    # pair10_epoch_10 --> lgbm: 0.79594
    # pair30_epoch_10 --> lgbm: 0.78262
    # pair100_epoch_10 --> lgbm: 0.76246
    # pair500_epoch_10 --> lgbm: 0.72548
    # pair10_epoch_40 --> lgbm: 0.75814
    # pair30_epoch_40 --> lgbm: 0.7651
    # pair100_epoch_40 --> lgbm: 0.79064
    # pair500_epoch_40 --> lgbm: 0.73704
    # toto10_epoch_10 --> lgbm: 0.66176
    # toto100_epoch_10 --> lgbm: 0.84092
    # toto500_epoch_10 --> lgbm: 0.74026
    # toto10_epoch_40 --> lgbm: 0.87402
    # toto30_epoch_40 --> lgbm: 0.8707
    # toto100_epoch_40 --> lgbm: 0.86226
    # toto500_epoch_40 --> lgbm: 0.70982
