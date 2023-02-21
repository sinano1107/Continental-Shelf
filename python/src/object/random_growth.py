# 四面体と選択済みの面が与えられる。
# 内心点の法線方向に成長点を作る。そのベクトルの長さrを出力する。
# =>増加した体積が報酬

import numpy as np
import matplotlib.pyplot as plt
from lib.generate_tetrahedron import generateTetrahedron
from lib.growth import growth


if __name__ == '__main__':
    episodes = 1000
    reward_history = []
    
    for _ in range(episodes):
        positions, normals = generateTetrahedron()
        
        # 面数
        meshes = len(positions) // 3
        
        # 面をランダムに選択
        select_index = np.random.choice(meshes)
        select_mesh = positions[select_index * 3 : (select_index + 1) * 3]
        normal_vector = normals[select_index * 3]
        
        # 成長させる長さ
        r = np.random.rand()
        
        # 成長
        reward = growth(select_mesh, normal_vector, r)
        reward_history.append(reward)
    
    plt.plot(range(episodes), reward_history)
    plt.show()