from fastapi import FastAPI
import numpy as np
from src.object.lib.generate_tetrahedron import generateTetrahedron
from src.object.lib.growth import growth
from src.object.lib.normalize import normalize
from src.object.model import ObjectModel


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


##########################################################

mesh_count = 30 # 偶数で指定してください
growth_count = (mesh_count - 4) // 2
obj_model: ObjectModel

# ニューラルネットワークによる確率的なオブジェクトの生成
@app.get('/generate')
def get_generate():
    global obj_model
    
    # modelを新たに生成
    obj_model = ObjectModel()
    
    # 結晶化
    positions, normals = obj_model.crystallization(growth_count)
    
    return {
        'positions': positions.tolist(),
        'normals': normals.tolist()
    }

# ニューラルネットワークの最適化（ユーザーが気に入ったときだけ呼ばれる）
@app.get('/update/{isGood}')
def get_learn(isGood: bool):
    # リアクションが良いものであれば学習する
    if isGood:
        obj_model.update()
    
    # 新たに結晶化
    positions, normals = obj_model.crystallization(growth_count)
    
    return {
        'positions': positions.tolist(),
        'normals': normals.tolist()
    }