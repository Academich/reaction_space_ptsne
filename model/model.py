import torch
from torch import nn











# class LinearClassificator(nn.Module):
#
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.lin = nn.Linear(dim_in, dim_out)
#
#     def forward(self, x):
#         return self.lin(x)
#
#
# def fit(model, train_ds, loss_func, optimizer, n_epochs, batch_size):
#     n_points = len(train_ds)
#     for epoch in range(n_epochs):
#         losses = []
#         for i in range(n_points // batch_size):
#             from_i = i * batch_size
#             to_i = from_i + min(batch_size, n_points - from_i)
#             x_batch, y_batch = train_ds[from_i:to_i]
#             preds = model(x_batch)
#             loss = loss_func(preds, y_batch)
#             losses.append(loss.item())
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         print(f"Epoch {epoch + 1}. Mean Loss: {torch.tensor(losses).mean()}")
