from numpy import array, ndarray
from numpy.linalg import norm


# 参考：https://examist.jp/mathematics/planar-vector/naisin-itivector/
def solve_incenter(a: ndarray, b: ndarray, c: ndarray):
    vector_ab = b - a
    vector_ac = c - a
    vector_bc = c - b
    
    norm_ab = norm(vector_ab)
    norm_ac = norm(vector_ac)
    norm_bc = norm(vector_bc)
    norm_sum = norm_ab + norm_ac + norm_bc
    
    incenter_vector_b = norm_ac / norm_sum * vector_ab
    incenter_vector_c = norm_ab / norm_sum * vector_ac
    return a + incenter_vector_b + incenter_vector_c


if __name__ == '__main__':
    # 二次元
    a = array([0, 0])
    b = array([14, 0])
    c = array([5, 12])
    print(solve_incenter(a, b, c))
    
    # 三次元
    a = array([2, 5, 3])
    b = array([-4, 2, -1])
    c = array([1, -3, 2])
    print(solve_incenter(a, b, c))