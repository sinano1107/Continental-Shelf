import numpy as np


def normalize(positions: np.ndarray):
    # positionsのmaxとminを算出
    max = positions.max(axis=0)
    min = positions.min(axis=0)
    # 各軸のmaxとminの絶対値を足すことでその軸の長さを算出
    distance = max + np.abs(min)
    # 最も大きかった軸の長さ
    max_distance = distance.max()
    # 最も大きかった軸の長さで割ることで1の範囲に収まるようにする
    normalized = positions / max_distance
    normalized_distance = distance / max_distance
    
    return normalized, max_distance, normalized_distance


if __name__ == '__main__':
    from generate_tetrahedron import generateTetrahedron
    positions, _ = generateTetrahedron()
    normalized, max_distance, normalized_distance = normalize(np.array(positions))
    print(normalized, max_distance, normalized_distance)