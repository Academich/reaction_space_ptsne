import matplotlib.pyplot as plt
from torch.nn import Module
from torch.utils.data import Dataset
from model.model import get_batch_embeddings
from config import config


def plot_embs(trained_model: Module,
              untrained_ref_model: Module,
              ds: Dataset,
              bs=config.training_params["batch_size"]):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for ref_embs_batch, labels_batch in get_batch_embeddings(untrained_ref_model, ds, bs):
        x_init = ref_embs_batch[:, 0]
        y_init = ref_embs_batch[:, 1]
        ax[0].scatter(x_init, y_init, c=labels_batch, s=2, cmap="hsv")
    for trained_embs_batch, labels_batch in get_batch_embeddings(trained_model, ds, bs):
        x = trained_embs_batch[:, 0]
        y = trained_embs_batch[:, 1]
        ax[1].scatter(x, y, c=labels_batch, s=2, cmap="hsv")
    ax[0].set_title("Before training")
    ax[1].set_title("After training")
    plt.suptitle("Final embedding space")
    plt.show()
