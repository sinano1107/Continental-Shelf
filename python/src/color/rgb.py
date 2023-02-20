import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(1, 128)
        self.l2 = nn.Linear(128, 3)
    
    def forward(self):
        input = torch.randn(1)
        x = F.relu(self.l1(input))
        x = torch.sigmoid(self.l2(x))
        return x


# 参考：https://qiita.com/shinido/items/2904fa1e9a6c78650b93
MAX = torch.sqrt(torch.tensor(3))
def difference(rgb, target):
    return torch.sqrt(((target - rgb) ** 2).sum()) / MAX


if __name__ == '__main__':
    epochs = 1000
    reset_epoch = 100
    
    pi = Policy()
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    target = torch.rand(3)
    losses = np.array([])
    colors = np.empty((0, 3))
    
    for epoch in range(epochs):
        rgb = pi.forward()
        loss = difference(rgb, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses = np.append(losses, loss.item())
        colors = np.append(colors, [rgb.detach().numpy()], axis=0)
        
        if epoch % reset_epoch == 0:
            target = torch.rand(3)
    
    plt.plot(range(epochs), losses)
    plt.bar(range(epochs), losses.max(), color=colors, width=1)
    plt.show()