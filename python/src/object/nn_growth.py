# 四面体と選択済みの面が与えられる。
# 内心点の法線方向に成長点を作る。そのベクトルの長さrを出力する。
# =>増加した体積が報酬

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from lib.generate_tetrahedron import generateTetrahedron
from lib.growth import growth
from lib.normalize import normalize
from lib.solve_volume import solve_volume


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(37, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 10)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x), dim=0)
        return x

if __name__ == '__main__':
    episodes = 1000
    loss_history = []
    reward_history = []
    
    # env
    positions, normals = generateTetrahedron()
    select_index = np.random.choice(4)
    normal_vector = normals[select_index * 3]
    pi = Policy()
    optimizer = optim.Adam(pi.parameters(), lr=0.0007)
    
    for episode in range(1, episodes + 1):
        first_tetrahedron = positions
        
        # 入力
        input = torch.tensor(np.append(select_index, positions))
        
        # 成長させる長さ
        probs = pi.forward(input)
        m = Categorical(probs)
        action = m.sample()
        r = action.item() * 0.1
        
        # 成長
        new_positions, _, new_tetrahedron = growth(positions, normals, select_index, r)
        
        # 報酬
        _, max_distance, _ = normalize(np.array(new_positions))
        first_tetrahedron /= max_distance
        new_tetrahedron /= max_distance
        reward = solve_volume(first_tetrahedron[0], first_tetrahedron[1], first_tetrahedron[2], first_tetrahedron[3])
        reward += solve_volume(new_tetrahedron[0], new_tetrahedron[1], new_tetrahedron[2], new_tetrahedron[3])
        
        # 損失
        loss = -m.log_prob(action) * reward
        
        # 学習
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 保存
        loss_history.append(loss.item())
        reward_history.append(reward)
        
        # 変化
        if episode % 100 == 0:
            select_index = np.random.choice(4)
            normal_vector = normals[select_index * 3]
    
    # プロット
    plt.title('mean = {}'.format(np.mean(reward_history)))
    plt.plot(range(episodes), loss_history)
    plt.plot(range(episodes), reward_history)
    plt.show()
    
    print(pi.forward(input))
