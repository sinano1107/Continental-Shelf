# 参考：https://algorithm.joho.info/programming/python/numpy-rotation-matrix/

import numpy as np
import matplotlib.pyplot as plt


def rotate_x(vector, rad):
    """x軸周りの回転"""
    C = np.cos(rad)
    S = np.sin(rad)
    R_x = np.matrix((
        (1, 0, 0),
        (0, C, -S),
        (0, S, C)
    ))
    res = np.dot(R_x, vector)
    return np.array([res[0,0], res[0,1], res[0,2]])


def rotate_y(vector, rad):
    """y軸周りの回転"""
    C = np.cos(rad)
    S = np.sin(rad)
    R_y = np.matrix((
        (C, 0, S),
        (0, 1, 0),
        (-S, 0, C)
    ))
    res = np.dot(R_y, vector)
    return np.array([res[0,0], res[0,1], res[0,2]])

def main():
    # 元の行列
    vector = np.array((1,1,0))
    # 回転行列の生成
    rotated = rotate_x(vector, np.pi / 2)
    rotated = rotate_y(rotated, np.pi / 2)
    
    # プロットの設定
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1 ,1])
    
    # 原点
    ax.scatter(0, 0, 0, color='b')
    
    # 元ベクトル
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, arrow_length_ratio=0.1)
    
    # 回転後のベクトル
    ax.quiver(0, 0, 0, rotated[0], rotated[1], rotated[2], length=1, arrow_length_ratio=0.1, color='orange')
    
    plt.show()


if __name__ == '__main__':
    main()