from torch import optim, device

from datasets import load_mnist_some_classes
from model.model import NeuralMapping, fit_model
from plot_embeddings import plot_embs
from config import config

import datetime

if __name__ == '__main__':
    # Defining dataset
    include_classes = (1, 3, 4, 5, 6)
    points, labels = load_mnist_some_classes(include_classes)

    # Defining instruments
    dev = device(config.dev)
    ffnn = NeuralMapping(dim_input=points.size(1)).to(dev)
    opt = optim.SGD(ffnn.parameters(), **config.optimization_conf)

    start = datetime.datetime.now()
    # Training and evaluating
    init_embs = ffnn(points.to(dev)).cpu().detach().numpy()
    fit_model(ffnn, points.to(dev), opt, **config.training_params)
    fin = datetime.datetime.now()
    ffnn.eval()
    final_embs = ffnn(points.to(dev)).cpu().detach().numpy()

    print("time elapsed:", fin - start)
    plot_embs(init_embs, final_embs, labels.numpy())
