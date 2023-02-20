# 色を学習するモデルの開発

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
        self.l2 = nn.Linear(128, 1)
    
    def forward(self):
        input = torch.randn(1)
        x = F.relu(self.l1(input))
        x = F.sigmoid(self.l2(x))
        return x


if __name__ == '__main__':
    pi = Policy()
    optimizer = optim.Adam(pi.parameters(), lr=0.0002)
    mserror = nn.MSELoss()
    losses = np.array([])
    
    target = torch.tensor([1.0])
    
    for epoch in range(1000):
        res = pi.forward()
        
        loss = mserror(res, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses = np.append(losses, loss.item())
        
        # 途中でターゲットを変える
        if epoch == 500:
            target = torch.tensor([0.0])
    
    plt.plot(range(len(losses)), losses)
    plt.show()