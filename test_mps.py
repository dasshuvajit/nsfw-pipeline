import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

x = torch.rand(1000, 1000).to(device)
y = torch.matmul(x, x)
print(y.device)