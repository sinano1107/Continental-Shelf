# 四面体と選択済みの面が与えられる。
# 内心点の法線方向に成長点を作る。そのベクトルの長さrを出力する。
# =>増加した体積が報酬

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.generate_tetrahedron import generateTetrahedron
from lib.growth import growth
from lib.normalize import normalize


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(37, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

if __name__ == '__main__':
    episodes = 10000
    loss_history = []
    reward_history = []
    
    pi = Policy()
    optimizer = optim.Adam(pi.parameters(), lr=0.002)
    
    for episode in range(1, episodes + 1):
        positions, normals = generateTetrahedron()
        
        # 面数
        meshes = len(positions) // 3
        
        # 面をランダムに選択
        select_index = np.random.choice(meshes)
        normal_vector = normals[select_index * 3]
        
        # 入力
        input = torch.tensor(np.append(select_index, positions))
        
        # 成長させる長さ
        r = pi.forward(input)
        
        # 成長
        positions, normal = growth(positions, normals, select_index, r.item())
        
        # 報酬
        _, _, normalized_distance = normalize(np.array(positions))
        reward = normalized_distance.sum() / 3
        
        # 損失
        loss = -torch.log(r) * reward
        
        # 学習
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 保存
        loss_history.append(loss.item())
        reward_history.append(reward)
        
        # ログ
        print('\repisode: {}, loss: {:.2f}, reward: {:.2f}'.format(episode, loss.item(), reward), end='')
    
    # プロット
    plt.title('mean = {}'.format(np.mean(reward_history)))
    plt.plot(range(episodes), loss_history)
    plt.plot(range(episodes), reward_history)
    plt.show()
