import torch.nn.functional as F
from torch import optim, tensor, matmul, sum, ones, cat, zeros
from torch.utils.data import TensorDataset

from datasets import load_mnist_some_classes
from utils import make_joint
from model.model import NeuralMapping, fit_model
from plot_embeddings import plot_embs

import datetime

if __name__ == '__main__':
    include_classes = (1, 3, 4, 5, 6)
    points, labels = load_mnist_some_classes(include_classes)
    start = datetime.datetime.now()
    ffnn = NeuralMapping(dim_input=points.size(1)).cuda()
    opt = optim.SGD(ffnn.parameters(), lr=0.05, momentum=0.9)
    fit_model(ffnn, points.cuda(), opt, perplexity=100, n_epochs=10, batch_size=200)
    fin = datetime.datetime.now()
    print("time elapsed:", fin - start)
    ffnn.eval()
    final_embs = ffnn(points.cuda()).cpu().detach().numpy()
    plot_embs(final_embs, labels.numpy())


