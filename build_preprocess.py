import multiprocessing as mp
import geopandas as gpd
import pyvista as pv
import numpy as np
from tqdm import tqdm
from functools import partial
from pybdshadow.utils import make_clockwise

def generate_mesh(building, resolution=10):
    """生成单个建筑物的3D模型（子进程用）"""
    all_points = []
    all_faces = []
    num_p = 0
    
    for _, row in building.iterrows():
        polygon = row['geometry']
        height = row['height']
        
        # 修复无效几何
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygon = make_clockwise(polygon)
        
        # 生成屋顶和墙面
        x, y = polygon.exterior.xy
        base_vertices = np.column_stack([x, y, np.zeros(len(x))])
        roof_vertices = np.column_stack([x, y, np.full(len(x), height)])
        
        # 屋顶
        roof_face = [len(roof_vertices)] + list(range(len(roof_vertices)))
        roof = pv.PolyData(roof_vertices, faces=[roof_face]).triangulate()
        roof = roof.subdivide_adaptive(max_edge_len=resolution, max_tri_area=resolution**2)
        
        # 墙面
        for i in range(len(base_vertices)-1):
            wall_vertices = np.array([
                base_vertices[i], base_vertices[i+1],
                roof_vertices[i+1], roof_vertices[i]
            ])
            wall_faces = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
            wall = pv.PolyData(wall_vertices, faces=wall_faces)
            wall = wall.subdivide_adaptive(max_edge_len=resolution, max_tri_area=resolution**2)
            
            # 合并到总网格
            for mesh in [roof, wall]:
                # 创建新的数组而不是直接修改只读数组
                faces = np.copy(mesh.faces.reshape(-1, 4))
                faces[:, 1:] += num_p
                all_points.append(mesh.points)
                all_faces.append(faces)
                num_p += len(mesh.points)
    
    return pv.PolyData(np.vstack(all_points), np.vstack(all_faces)) if all_points else None

def combine_meshes(meshes):
    """高效合并多个PolyData"""
    all_points, all_faces, offset = [], [], 0
    for mesh in meshes:
        if mesh is None:
            continue
        all_points.append(mesh.points)
        # 创建新的数组而不是直接修改只读数组
        faces = np.copy(mesh.faces.reshape(-1, 4))
        faces[:, 1:] += offset
        all_faces.append(faces)
        offset += len(mesh.points)
    return pv.PolyData(np.vstack(all_points), np.vstack(all_faces)) if all_points else None

def generate_mesh_parallel(buildings, resolution=10, n_jobs=None):
    """并行生成3D模型"""
    n_jobs = n_jobs or mp.cpu_count()
    chunks = np.array_split(buildings, n_jobs)
    
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(partial(generate_mesh, resolution=resolution), chunks),
            total=len(chunks),
            desc="Processing buildings"
        ))
    
    return combine_meshes(results)

# 使用示例
if __name__ == "__main__":
    buildings = gpd.read_file('shenzhen_building.geojson')
    mesh = generate_mesh_parallel(buildings, resolution=10, n_jobs=24)
    if mesh:
        mesh.save('buildings_3d.vtk')
    else:
        print("No valid buildings processed.")