from fastapi import FastAPI
import numpy as np
from src.object.lib.generate_tetrahedron import generateTetrahedron
# from src.object.lib.growth import growth
# from src.object.lib.normalize import normalize

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


# @app.get('/growth')
# def get_growth():
#     global positions, normals
#     positions, normals, _ = growth(np.array(positions), normals, len(positions) // 3 - 1, 1)
#     positions, _, _ = normalize(np.array(positions))
#     return {
#         'positions': positions,
#         'normals': [n.tolist() for n in normals]
#     }