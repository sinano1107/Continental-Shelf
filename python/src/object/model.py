import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from .lib.generate_tetrahedron import generateTetrahedron
from .lib.growth import growth
from .lib.normalize import normalize


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1, 128)
        self.l2 = nn.Linear(128, 128)
        
        self.r_mean = nn.Linear(128, 1)
        self.r_log_std = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.r_prob = torch.tensor([])
    
    def forward(self):
        input = torch.randn(1)
        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))
        
        r_mean = self.r_mean(x)
        r_log_std = self.r_log_std(x)
        r_log_std = torch.clamp(r_log_std, 0, 0)
        
        return r_mean, r_log_std

    def sample(self, n):
        """モデルに基づきサンプリングを行う
        Parameters
            n: サンプルする数
        """
        r_mean, r_log_std = self.forward()
        
        r_std = r_log_std.exp()
        
        normal = Normal(r_mean, r_std)
        r = normal.sample_n(n)
        r_prob = normal.log_prob(r)
        r = torch.clamp(r, 0, 1)
        
        return r, r_prob
    
    def crystallization(self, growth_count):
        """結晶化を行う"""
        positions, normals = generateTetrahedron()
        
        # 成長させる長さの算出
        r, self.r_prob = self.sample(growth_count)
        
        # 成長
        for i in range(growth_count):
            select_mesh_index = np.random.choice(len(positions) // 3)
            positions, normals, _ = growth(positions, normals, select_mesh_index, r[i].item(), 0, 0)
            # 正規化
            positions, _, _ = normalize(positions)
        
        return positions, normals
    
    def update(self):
        """モデルを更新する"""
        loss = -self.r_prob.sum() / len(self.r_prob)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()