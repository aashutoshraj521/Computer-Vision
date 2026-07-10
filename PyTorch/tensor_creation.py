'''When writing PyTorch code, make it a reflex to inspect tensors immediately after creating or loading them:
print(x.shape)
print(x.dtype)
print(x.device)
Many debugging sessions that could take hours are resolved in seconds by checking these three properties. This habit becomes even more valuable once we start implementing CNNs and reproducing research papers.'''

import torch

# Scalar (0-D)
scalar = torch.tensor(7)

# Vector (1-D)
vector = torch.tensor([2, 5, 8, 1])

# Matrix (2-D)
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# 3-D Tensor
tensor3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9,10], [11,12]]
])

print("Shape of scalar: ", scalar.shape)
print("Shape of vector: ", vector.shape)
print("Shape of matrix: ", matrix.shape)
print("Shape of tensor: ", tensor3d.shape)

# Many bugs are solved just by inspecting these three properties.
print("Shape of tensor: ", tensor3d.shape)
print("Tensor data type: ", tensor3d.dtype)
print("Tensor device: ", tensor3d.device)

# Different ways to create tensors
a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.ones((2, 3))
d = torch.rand((2, 3))
e = torch.arange(0, 10)

# Inspect one tensor
print(d)
print("Tensor:\n", d)
print("Shape :", d.shape)
print("Data Type :", d.dtype)
print("Device :", d.device)