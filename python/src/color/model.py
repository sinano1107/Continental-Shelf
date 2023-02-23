import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ColorModel(nn.Module):
    def __init__(self):
        super(ColorModel, self).__init__()
        self.l1 = nn.Linear(1, 128)
        self.l2 = nn.Linear(128, 128)
        
        # red
        self.r_mean = nn.Linear(128, 1)
        self.r_log_std = nn.Linear(128, 1)
        # green
        self.g_mean = nn.Linear(128, 1)
        self.g_log_std = nn.Linear(128, 1)
        # blue
        self.b_mean = nn.Linear(128, 1)
        self.b_log_std = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.r_prob = torch.tensor([])
        self.g_prob = torch.tensor([])
        self.b_prob = torch.tensor([])
    
    def forward(self):
        input = torch.randn(1)
        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))

        # red
        r_mean = self.r_mean(x)
        r_log_std = self.r_log_std(x)
        r_log_std = torch.clamp(r_log_std, 0, 0)
        # green
        g_mean = self.g_mean(x)
        g_log_std = self.g_log_std(x)
        g_log_std = torch.clamp(g_log_std, 0, 0)
        # blue
        b_mean = self.g_mean(x)
        b_log_std = self.b_log_std(x)
        b_log_std = torch.clamp(b_log_std, 0, 0)
        
        return r_mean, r_log_std, g_mean, g_log_std, b_mean, b_log_std
    
    def sample(self):
        """モデルに基づきサンプリングを行う"""
        r_mean, r_log_std, g_mean, g_log_std, b_mean, b_los_std = self.forward()
        
        r_std = r_log_std.exp()
        g_std = g_log_std.exp()
        b_std = b_los_std.exp()
        
        # red
        normal = Normal(r_mean, r_std)
        r = normal.sample()
        r_prob = normal.log_prob(r)
        r = torch.clamp(r, 0, 1)
        # green
        normal = Normal(g_mean, g_std)
        g = normal.sample()
        g_prob = normal.log_prob(g)
        g = torch.clamp(g, 0, 1)
        # blue
        normal = Normal(b_mean, b_std)
        b = normal.sample()
        b_prob = normal.log_prob(b)
        b = torch.clamp(b, 0, 1)
        
        return r, r_prob, g, g_prob, b, b_prob
    
    def generate(self):
        """色を出力する"""
        r, self.r_prob, g, self.g_prob, b, self.b_prob = self.sample()
        return [r.item(), g.item(), b.item()]
    
    def update(self):
        """モデルを更新する"""
        loss = -(self.r_prob * self.g_prob * self.b_prob)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()