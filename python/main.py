from fastapi import FastAPI
from src.object.generate_tetrahedron import generateTetrahedron

app = FastAPI()


@app.get('/')
def root():
    positions, normals = generateTetrahedron()
    return {
        'positions': [p.tolist() for p in positions],
        'normals': [n.tolist() for n in normals]
    }