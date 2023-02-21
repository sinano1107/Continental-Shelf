import numpy as np
from numpy import ndarray, array


# 参考：https://hiraocafe.com/note/tetrahedron_volume.html
def solve_volume(a: ndarray, b: ndarray, c: ndarray, d: ndarray):
    '''与えられた4点で構成される四面体の体積を求めます'''
    
    # a,b,cを通る平面の連立方程式のx,y,z要素
    cross = np.cross(b - a, c - a)
    
    # 三角形abcの面積
    S = np.linalg.norm(cross) / 2
    
    # 平面とdの距離
    L = abs((cross * d).sum()) / np.sqrt((cross ** 2).sum(dtype=float))
    
    return S * L / 3


if __name__ == '__main__':
    a = array([0, 0, 0])
    b = array([6, 0, -2])
    c = array([0, 6, 3])
    d = array([5, -1, 6])
    
    V = solve_volume(a, b, c, d)
    print(V)