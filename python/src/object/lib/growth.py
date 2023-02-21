import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from incenter import solve_incenter
else:
    from .incenter import solve_incenter


def growth(positions: np.ndarray, normals: np.ndarray, select_mesh_index: int, r: float, render=False):
    '''面を成長させて、その体積を返す。'''
    positions = positions.copy()
    normals = normals.copy()
    
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
    positions = np.delete(positions, [select_start, select_start + 1, select_start + 2], axis=0)
    normals = np.delete(normals, [select_start, select_start + 1, select_start + 2], axis=0)
    # del positions[select_start : select_end]
    # del normals[select_start : select_end]
    
    # 成長させた四面体
    growth_tetrahedron = [growth_point]
    
    # 新たなメッシュと法線を追加
    for i, p1 in enumerate(select_mesh):
        growth_tetrahedron.append(p1)
        
        p2 = select_mesh[(i + 1) % 3]
        positions = np.append(positions, [growth_point, p1, p2], axis=0)
        cross = np.cross(p1 - growth_point, p2 - p1)
        normal = cross / np.linalg.norm(cross)
        normals = np.append(normals, [normal, normal, normal], axis=0)
    
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
    
    return positions, normals, growth_tetrahedron


if __name__ == '__main__':
    from generate_tetrahedron import generateTetrahedron
    from normalize import normalize
    positions, normals = generateTetrahedron()
    positions, normals, _ = growth(positions, normals, 0, 1)
    _, _, normalized_distance = normalize(np.array(positions))
    print(normalized_distance)