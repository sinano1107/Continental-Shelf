from fastapi import FastAPI
import numpy as np
from src.object.lib.generate_tetrahedron import generateTetrahedron
from src.object.lib.growth import growth
from src.object.lib.normalize import normalize

app = FastAPI()

positions = np.array([])
normals = np.array([])

@app.get('/tetrahedron')
def get_tetrahedron():
    global positions, normals
    positions, normals = generateTetrahedron()
    return {
        'positions': positions.tolist(),
        'normals': normals.tolist()
    }


@app.get('/growth')
def get_growth():
    global positions, normals
    
    # 成長させる面、長さ、回転を乱数で定める
    i = np.random.choice(len(positions) // 3)
    r = np.random.uniform(0, 1)
    rad_x, rad_y = np.random.uniform(-np.pi/2, np.pi/2, 2)
    
    positions, normals, _ = growth(positions, normals, i, r, rad_x, rad_y)
    positions, _, _ = normalize(positions)
    return {
        'positions': positions.tolist(),
        'normals': normals.tolist(),
    }