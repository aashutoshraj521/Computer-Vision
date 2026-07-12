import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2

y.backward()

print("\nx =", x)
print("y =", y)
print("Gradient =", x.grad)

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b
loss = y ** 2

loss.backward()

print("\nGradient of w:", w.grad)
print("Gradient of b:", b.grad)