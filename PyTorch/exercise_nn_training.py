import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([
            [1.,2.],
            [2.,1.],
            [4.,5.],
            [5.,4.]
        ])
        self.y = torch.tensor([0,0,1,1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return self.x[index], self.y[index]

dataset = ToyDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2,2)

    def forward(self,x):
        return self.fc(x)

model = SimpleClassifier()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):
    model.train()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

    print(epoch,loss.item())