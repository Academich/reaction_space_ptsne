import matplotlib.pyplot as plt


def plot_embs(initial_embs, final_embs, labels=None):
    x_init = initial_embs[:, 0]
    y_init = initial_embs[:, 1]
    x = final_embs[:, 0]
    y = final_embs[:, 1]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(x_init, y_init, c=labels, s=2, cmap="hsv")
    ax[0].set_title("Before training")
    ax[1].scatter(x, y, c=labels, s=2, cmap="hsv")
    ax[1].set_title("After training")
    plt.suptitle("Final embedding space")
    plt.show()
