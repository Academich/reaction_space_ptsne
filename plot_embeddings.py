import matplotlib.pyplot as plt


def plot_embs(final_embs, labels=None):
    x = final_embs[:, 0]
    y = final_embs[:, 1]
    plt.figure()
    plt.scatter(x, y, c=labels, s=2)
    plt.title("Пространство представлений t-SNE")
    plt.show()
