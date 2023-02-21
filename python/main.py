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
    positions, normals, _ = growth(positions, normals, 0, 0.1)
    positions, _, _ = normalize(positions)
    return {
        'positions': positions.tolist(),
        'normals': normals.tolist(),
    }