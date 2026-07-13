import torch
import torch.nn as nn

class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4,2)

    def forward(self,x):
        return self.fc(x)

model = SimpleNN()
x = torch.rand((1,4))
output = model(x)

print(output)