# 四面体と選択済みの面が与えられる。
# 内心点の法線方向に成長点を作る。そのベクトルの長さrを出力する。
# =>増加した体積が報酬

import numpy as np
import matplotlib.pyplot as plt
from lib.generate_tetrahedron import generateTetrahedron
from lib.growth import growth
from lib.normalize import normalize


if __name__ == '__main__':
    episodes = 10000
    reward_history = []
    
    for _ in range(episodes):
        positions, normals = generateTetrahedron()
        
        # 面数
        meshes = len(positions) // 3
        
        # 面をランダムに選択
        select_index = np.random.choice(meshes)
        normal_vector = normals[select_index * 3]
        
        # 成長させる長さ
        r = np.random.rand()
        
        # 成長
        positions, normals = growth(positions, normals, select_index, r)
        
        # 報酬
        _, _, normalized_distance = normalize(np.array(positions))
        reward = normalized_distance.sum() / 3
        
        # 保存
        reward_history.append(reward)
    
    plt.title('mean = {}'.format(np.mean(reward_history)))
    plt.plot(range(episodes), reward_history)
    plt.show()