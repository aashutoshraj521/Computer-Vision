# import torch

# device = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "cpu"
# )

# x = torch.rand((3,224,224))

# x = x.to(device)

# print(x.device)

print("--- Smoke Test: Script is executing! ---")

import torch

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Targeting device: {device}")

x = torch.rand((3, 224, 224))
x = x.to(device)
print(f"Tensor is successfully running on: {x.device}")