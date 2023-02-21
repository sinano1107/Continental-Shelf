import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from incenter import solve_incenter
else:
    from .incenter import solve_incenter


def growth(positions: list[np.ndarray], normals: list[np.ndarray], select_mesh_index: int, r: float, render=False):
    '''面を成長させて、その体積を返す。'''
    # 成長させるメッシュのpositionの範囲
    select_start = select_mesh_index * 3
    select_end = (select_mesh_index + 1) * 3
    
    # 成長させるメッシュ
    select_mesh = positions[select_start: select_end]
    
    # 内心
    incenter = solve_incenter(select_mesh[0], select_mesh[1], select_mesh[2])
    
    # 成長点
    normal_vector = normals[select_start]
    growth_point = incenter + normal_vector * r
    
    # positions,normalsから成長メッシュを削除
    del positions[select_start : select_end]
    del normals[select_start : select_end]
    
    # 新たなメッシュと法線を追加
    for i, p1 in enumerate(select_mesh):
        p2 = select_mesh[(i + 1) % 3]
        positions.extend([growth_point, p1, p2])
        cross = np.cross(p1 - growth_point, p2 - p1)
        normal = cross / np.linalg.norm(cross)
        normals.extend([normal, normal, normal])
    
    # 表示
    # ! matpplotlibは少し歪むので注意
    if render:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((1, 1, 1))
        select_mesh = np.array(select_mesh)
        
        # 頂点
        ax.scatter(select_mesh[:,0], select_mesh[:,1], select_mesh[:,2], color='b')
        # 内心
        ax.scatter(incenter[0], incenter[1], incenter[2], color='r')
        # 成長点
        ax.scatter(growth_point[0], growth_point[1], growth_point[2], color='black')
        
        plt.show()
    
    return positions, normals


if __name__ == '__main__':
    from generate_tetrahedron import generateTetrahedron
    from normalize import normalize
    positions, normals = generateTetrahedron()
    positions, normals = growth(positions, normals, 0, 1)
    _, _, normalized_distance = normalize(np.array(positions))
    print(normalized_distance)