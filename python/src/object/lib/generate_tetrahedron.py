import random # type: ignore
import numpy as np


def generateTetrahedron() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ランダムな三角錐を生成します"""
    positions = []
    for _ in range(4):
        # -1~1の乱数を3つ持った配列を追加
        positions.append(np.array([random.uniform(-1, 1) for _ in range(3)]))
    
    return connect2Tetrahedron(positions)


def connect2Tetrahedron(p: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """与えられた４点を結びます"""
    assert len(p) == 4, '値が４つの配列を渡してください'
    
    positions = []
    normals = []
    
    # region 最初の面の向きを確定
    # 1->2->3の順で結んだ場合の法線
    normalVector = np.cross(p[1] - p[0], p[2] - p[1])
    # 正規化
    normalVector = normalVector / np.linalg.norm(normalVector)
    
    # p1->p4のベクトル
    vector1to4 = p[3] - p[0]
    # 正規化
    vector1to4 = vector1to4 / np.linalg.norm(vector1to4)
    
    # normalVectorとvector1to4の内積
    # 正の値の時、同じ方向を向いているため1->2->3の結び方は正しくない
    # 負の時、別方向を向いているため1->2->3の結び方で正しい
    theta = np.dot(normalVector, vector1to4)
    
    if theta < 0:
        # 正しいためそのまま代入
        positions.extend([p[0], p[1], p[2]])
        normals.extend([normalVector for _ in range(3)])
    else:
        # 正しくないため反転して代入
        positions.extend([p[0], p[2], p[1]])
        normals.extend([-normalVector for _ in range(3)])
    # endregion
    
    # region 残り3面を確定
    for index, pos_a in enumerate(positions.copy()):
        pos_b = positions[(index + 2) % 3]
        positions.extend([p[3], pos_a, pos_b])
        normal = np.cross(pos_a - p[3], pos_b - pos_a)
        normals.extend([normal for _ in range(3)])
    # endregion
    
    return (positions, normals)


if __name__ == '__main__':
    res = generateTetrahedron()
    print(res)