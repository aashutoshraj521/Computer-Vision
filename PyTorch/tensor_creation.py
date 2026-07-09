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

print(scalar.shape)
print(vector.shape)
print(matrix.shape)
print(tensor3d.shape)