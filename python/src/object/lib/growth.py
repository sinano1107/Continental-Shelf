import numpy as np
import matplotlib.pyplot as plt

if __name__ != '__main__':
    from .incenter import solve_incenter
    from .solve_volume import solve_volume


def growth(select_mesh: np.ndarray, normal_vector: np.ndarray, r: float, render=False):
    '''面を成長させて、その体積を返す。'''
    incenter = solve_incenter(select_mesh[0], select_mesh[1], select_mesh[2])
    growth_point = incenter + normal_vector * r
    volume = solve_volume(select_mesh[0], select_mesh[1], select_mesh[2], growth_point)
    
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
        
        plt.title('volume = {}'.format(volume))
        plt.show()
    
    return volume