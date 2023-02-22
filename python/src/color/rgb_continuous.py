import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(1, 128)
        self.l2 = nn.Linear(128, 128)
        
        self.red_mean_linear = nn.Linear(128, 1)
        self.red_log_std_linear = nn.Linear(128, 1)
        
        self.green_mean_linear = nn.Linear(128, 1)
        self.green_log_std_linear = nn.Linear(128, 1)
        
        self.blue_mean_linear = nn.Linear(128, 1)
        self.blue_log_std_linear = nn.Linear(128, 1)
    
    def forward(self):
        input = torch.randn(1)
        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))
        
        red_mean = self.red_mean_linear(x)
        red_log_std = self.red_log_std_linear(x)
        red_log_std = torch.clamp(red_log_std, 0, 0)
        
        green_mean = self.green_mean_linear(x)
        green_log_std = self.green_log_std_linear(x)
        green_log_std = torch.clamp(green_log_std, 0, 0)
        
        blue_mean = self.blue_mean_linear(x)
        blue_log_std = self.blue_log_std_linear(x)
        blue_log_std = torch.clamp(blue_log_std, 0, 0)
        
        return red_mean, red_log_std, green_mean, green_log_std, blue_mean, blue_log_std
    
    def sample(self):
        red_mean, red_log_std, green_mean, green_log_std, blue_mean, blue_log_std = self.forward()
        
        red_std = red_log_std.exp()
        green_std = green_log_std.exp()
        blue_std = blue_log_std.exp()
        
        normal = Normal(red_mean, red_std)
        red = normal.sample()
        red_prob = normal.log_prob(red)
        red = torch.clamp(red, 0, 1)
        
        normal = Normal(green_mean, green_std)
        green = normal.sample()
        green_prob = normal.log_prob(green)
        green = torch.clamp(green, 0, 1)
        
        normal = Normal(blue_mean, blue_std)
        blue = normal.sample()
        blue_prob = normal.log_prob(blue)
        blue = torch.clamp(blue, 0, 1)
        
        return red, red_prob, green, green_prob, blue, blue_prob


# 参考：https://qiita.com/shinido/items/2904fa1e9a6c78650b93
MAX = torch.sqrt(torch.tensor(3))
def difference(rgb, target):
    return torch.sqrt(((target - rgb) ** 2).sum()) / MAX


if __name__ == '__main__':
    torch.manual_seed(117)
    epochs = 1000
    
    pi = Policy()
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    target = torch.rand(3)
    losses = np.array([])
    colors = np.empty((0, 3))
    
    for epoch in range(epochs):
        r, r_prob, g, g_prob, b, b_prob = pi.sample()
        rgb = torch.tensor([r.item(), g.item(), b.item()])
        
        dif = difference(rgb, target)
        # 仮想的な報酬をdifから算出
        reward = dif <= 0.2

        if reward:
            loss = -(r_prob * g_prob * b_prob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            losses = np.append(losses, loss.item())
        else:
            losses = np.append(losses, 0)
        
        colors = np.append(colors, [rgb.detach().numpy()], axis=0)
        
        print('\repoch: {}'.format(epoch + 1), end='')
    
    plt.title('target = {}'.format(target), color=target.detach().numpy())
    plt.plot(range(epochs), losses, lw=4, label='losses')
    plt.bar(range(epochs), losses.max(), color=colors, width=1)
    plt.bar(range(epochs), losses.min(), color=colors, width=1)
    plt.legend()
    plt.show()