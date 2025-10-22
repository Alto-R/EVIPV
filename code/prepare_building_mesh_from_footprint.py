"""
从建筑footprint数据转换为建筑mesh
用于车顶光伏阴影计算

数据要求:
- 输入格式: GeoJSON, Shapefile, GeoPackage等GeoDataFrame支持的格式
- 必需字段: geometry (Polygon), height (数值)
- 坐标系: WGS84 (EPSG:4326)
"""

import trimesh
import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

# 导入RealSceneDL库的footprint转换功能
from RealSceneDL.footprint import footprint_to_tmesh, footprint_preprocess

# ============================================================================
# 配置区域 - 在此修改参数
# ============================================================================
CONFIG = {
    # 输入文件路径
    'FOOTPRINT_PATH': '../data/shenzhen_building.geojson',  # 建筑footprint数据路径 (GeoJSON/Shapefile/GeoPackage)

    # 输出文件路径
    'OUTPUT_MESH_PATH': '../data/shenzhen_building_mesh.ply',  # 输出mesh文件路径 (.ply格式，trimesh原生)

    # 网格参数
    'GRID_SIZE': None,  # mesh网格细分精度(米)，None表示不细分，建议值: 10-50米

    # 并行处理参数
    'N_JOBS': 20,  # 并行进程数: -1=使用所有CPU核心, 1=单进程(不并行), 2,4,8...=指定进程数
    'BATCH_SIZE': None,  # 每批建筑数量，None=自动根据CPU核心数分配
}
# ============================================================================


def footprint_preprocess_parallel(building_gdf, n_jobs=-1, batch_size=None):
    """
    并行处理版本的footprint_preprocess

    将GeoDataFrame分批并行处理，然后合并结果

    Parameters
    ----------
    building_gdf : geopandas.GeoDataFrame
        建筑footprint数据
    n_jobs : int, optional
        并行进程数。-1表示使用所有CPU核心，1表示单进程（不并行）
    batch_size : int, optional
        每批处理的建筑数量。None表示自动根据CPU核心数分配

    Returns
    -------
    geopandas.GeoDataFrame
        预处理后的建筑数据
    """
    import time

    n_buildings = len(building_gdf)

    # 确定进程数
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    # 如果建筑数量太少或只用单进程，直接调用原函数
    if n_buildings < 100 or n_jobs == 1:
        print(f"   使用单进程处理（建筑数: {n_buildings:,}）")
        return footprint_preprocess(building_gdf)

    # 确定批次大小
    if batch_size is None:
        batch_size = max(100, n_buildings // (n_jobs * 2))  # 每个进程至少处理2批

    # 计算批次数
    n_batches = int(np.ceil(n_buildings / batch_size))

    print(f"   并行处理配置:")
    print(f"     - 进程数: {n_jobs}")
    print(f"     - 批次大小: {batch_size:,} 建筑/批")
    print(f"     - 批次数: {n_batches}")

    # 分批
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_buildings)
        batch = building_gdf.iloc[start_idx:end_idx].copy()
        batches.append(batch)

    # 并行处理
    print(f"   ⚡ 开始并行处理...")
    start_time = time.time()

    try:
        # 使用joblib并行处理
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(footprint_preprocess)(batch)
            for batch in tqdm(batches, desc="   处理进度", ncols=80)
        )

        # 合并结果
        print(f"   🔗 合并结果...")
        processed_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

        # 重新分配building_id以保持连续性
        processed_gdf['building_id'] = range(len(processed_gdf))

        elapsed_time = time.time() - start_time

        print(f"   ✅ 并行处理完成")
        print(f"      - 耗时: {elapsed_time:.1f} 秒")
        print(f"      - 处理速度: {n_buildings/elapsed_time:.0f} 建筑/秒")

        return processed_gdf

    except Exception as e:
        print(f"   ⚠️  并行处理失败，回退到单进程: {e}")
        return footprint_preprocess(building_gdf)


def _extrude_batch(batch_buildings):
    """
    批量拉伸建筑的辅助函数（用于并行处理）

    Parameters
    ----------
    batch_buildings : geopandas.GeoDataFrame
        一批建筑数据

    Returns
    -------
    pyvista.PolyData
        合并后的mesh
    """
    merged_mesh = pv.MultiBlock()
    for idx, row in batch_buildings.iterrows():
        polygon = row['geometry']
        height = row['height']
        # 获取外部边界的坐标点，并移除最后一个重复点
        x, y = polygon.exterior.xy
        points = np.column_stack((x[:-1], y[:-1], np.zeros(len(x)-1)))
        face_a = [len(points)]+list(range(len(points)))
        faces = [face_a]
        mesh = pv.PolyData(points, faces)
        mesh = mesh.extrude((0, 0, height), capping=True)
        merged_mesh.append(mesh)

    # 合并这批建筑
    batch_mesh = merged_mesh.combine(merge_points=True)
    return batch_mesh


def footprint_to_tmesh_parallel(building_gdf, grid_size=None, n_jobs=-1, batch_size=None):
    """
    并行版本的 footprint_to_tmesh

    当 grid_size=None 时，使用并行化的拉伸方法显著提升性能

    Parameters
    ----------
    building_gdf : geopandas.GeoDataFrame
        建筑footprint数据
    grid_size : float, optional
        网格细分精度(米)，None表示不细分
    n_jobs : int, optional
        并行进程数。-1表示使用所有CPU核心
    batch_size : int, optional
        每批处理的建筑数量。None表示自动分配

    Returns
    -------
    trimesh.Trimesh
        建筑mesh对象
    pd.DataFrame
        三角形信息
    """
    import time
    from pyproj import CRS, Transformer

    # 如果需要细分，使用原始函数（已经优化过）
    if grid_size is not None:
        return footprint_to_tmesh(building_gdf, grid_size=grid_size)

    # 并行化拉伸过程
    n_buildings = len(building_gdf)

    # 转为投影坐标系
    lon = building_gdf['geometry'].iloc[0].centroid.x
    lat = building_gdf['geometry'].iloc[0].centroid.y
    project_epsg = CRS.from_proj4(
        "+proj=aeqd +lat_0="+str(lat)+" +lon_0="+str(lon)+" +datum=WGS84")
    building_gdf = building_gdf.to_crs(project_epsg)

    # 简化建筑
    building_gdf['geometry'] = building_gdf.simplify(0.001)

    # 确定进程数
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    # 确定批次大小
    if batch_size is None:
        batch_size = max(100, n_buildings // (n_jobs * 4))  # 每个进程处理4批

    # 计算批次数
    n_batches = int(np.ceil(n_buildings / batch_size))

    print(f"   并行拉伸配置:")
    print(f"     - 进程数: {n_jobs}")
    print(f"     - 批次大小: {batch_size:,} 建筑/批")
    print(f"     - 批次数: {n_batches}")

    # 分批
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_buildings)
        batch = building_gdf.iloc[start_idx:end_idx].copy()
        batches.append(batch)

    # 并行拉伸
    print(f"   ⚡ 开始并行拉伸建筑...")
    start_time = time.time()

    try:
        # 使用joblib并行处理每批建筑
        batch_meshes = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_extrude_batch)(batch)
            for batch in tqdm(batches, desc="   拉伸进度", ncols=80)
        )

        # 合并所有批次的mesh
        print(f"   🔗 合并所有mesh...")
        final_multiblock = pv.MultiBlock()
        for batch_mesh in batch_meshes:
            final_multiblock.append(batch_mesh)

        pyvista_mesh = final_multiblock.combine(merge_points=True)
        pyvista_mesh = pyvista_mesh.extract_surface().triangulate()
        pyvista_mesh.flip_normals()

        elapsed_time = time.time() - start_time
        print(f"   ✅ 并行拉伸完成")
        print(f"      - 耗时: {elapsed_time:.1f} 秒")
        print(f"      - 处理速度: {n_buildings/elapsed_time:.0f} 建筑/秒")

    except Exception as e:
        print(f"   ⚠️  并行处理失败，回退到单进程: {e}")
        import traceback
        traceback.print_exc()
        # 回退到原始函数
        return footprint_to_tmesh(building_gdf, grid_size=None)

    # 转换为trimesh
    vertices = pyvista_mesh.points
    faces_pyvista = pyvista_mesh.faces

    if pyvista_mesh.is_all_triangles:
        faces = faces_pyvista.reshape(-1, 4)[:, 1:4]
    else:
        print("警告：PyVista 网格包含非三角形面。Trimesh 可能会尝试对其进行三角化。")
        faces_pyvista_triangulated = pyvista_mesh.triangulate().faces
        faces = faces_pyvista_triangulated.reshape(-1, 4)[:, 1:4]

    # 坐标转换到 EPSG:4978
    target_crs = CRS("EPSG:4978")
    transformer = Transformer.from_crs(project_epsg, target_crs, always_xy=True)
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    z_coords = vertices[:, 2]

    x_transformed, y_transformed, z_transformed = transformer.transform(x_coords, y_coords, z_coords)
    vertices = np.stack((x_transformed, y_transformed, z_transformed), axis=-1)

    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh_mesh.visual = trimesh.visual.ColorVisuals(mesh=trimesh_mesh)
    trimesh_mesh.visual.face_colors = (np.array([1,1,1,1])*255).astype(int).tolist()

    # 坐标变换
    trimesh_mesh.apply_transform([
        [1,0,0,0],
        [0,0,1,0],
        [0,-1,0,0],
        [0,0,0,1]
    ])

    tri_info = pd.DataFrame()  # 简单模式没有详细的三角形信息

    return trimesh_mesh, tri_info

  
def prepare_building_mesh_from_footprint(
    footprint_path,
    output_mesh_path,
    grid_size=None,
    n_jobs=-1,
    batch_size=None
):
    """
    从建筑footprint数据转换为trimesh用于光伏计算

    Parameters
    ----------
    footprint_path : str
        建筑footprint数据路径，支持GeoJSON, Shapefile, GeoPackage等格式
    output_mesh_path : str
        输出mesh文件路径(.ply格式，trimesh原生格式)
    grid_size : float, optional
        mesh网格细分精度(米)。如果指定，会对屋顶和墙面进行精细网格划分，
        适用于高精度太阳能分析。默认None(不细分)
    n_jobs : int, optional
        并行进程数。-1表示使用所有CPU核心，1表示单进程（不并行）。默认-1
    batch_size : int, optional
        每批处理的建筑数量。None表示自动根据CPU核心数分配。默认None

    Returns
    -------
    trimesh.Trimesh
        建筑物mesh对象

    Raises
    ------
    FileNotFoundError
        如果footprint文件不存在
    ValueError
        如果footprint数据缺少必需字段(geometry, height)

    """
    print("="*60)
    print("🏗️  从建筑Footprint转换建筑mesh")
    print("="*60)

    footprint_path = Path(footprint_path)
    if not footprint_path.exists():
        raise FileNotFoundError(f"Footprint文件不存在: {footprint_path}")

    print(f"\n📂 输入: {footprint_path}")
    print(f"📂 输出: {output_mesh_path}")
    if grid_size:
        print(f"🔲 网格精度: {grid_size} 米")

    # 步骤1: 加载footprint数据
    print("\n🔄 正在加载建筑footprint数据...")
    try:
        building_gdf = gpd.read_file(str(footprint_path))
        print(f"✅ 成功加载 {len(building_gdf)} 个建筑footprint")
    except Exception as e:
        raise RuntimeError(f"加载footprint数据失败: {e}")

    # 检查必需字段
    required_fields = ['geometry', 'height']
    missing_fields = [f for f in required_fields if f not in building_gdf.columns]
    if missing_fields:
        raise ValueError(f"Footprint数据缺少必需字段: {missing_fields}")

    # 显示数据信息
    print(f"   建筑数量: {len(building_gdf):,}")
    print(f"   坐标系: {building_gdf.crs}")
    print(f"   高度范围: {building_gdf['height'].min():.1f}m - {building_gdf['height'].max():.1f}m")

    # 步骤2: 预处理footprint数据
    print("\n🔄 预处理建筑数据...")
    try:
        building_gdf = footprint_preprocess_parallel(building_gdf, n_jobs=n_jobs, batch_size=batch_size)
        print(f"✅ 预处理完成，保留 {len(building_gdf)} 个有效建筑")
    except Exception as e:
        print(f"⚠️  预处理出现问题: {e}")
        print("   尝试继续处理...")

    # 步骤3: 转换为trimesh（使用并行版本）
    print("\n🔄 转换为3D mesh...")
    try:
        building_trimesh, tri_info = footprint_to_tmesh_parallel(
            building_gdf,
            grid_size=grid_size,
            n_jobs=n_jobs,
            batch_size=batch_size
        )
        print(f"✅ 成功生成mesh")
        print(f"   顶点数: {len(building_trimesh.vertices):,}")
        print(f"   三角形数: {len(building_trimesh.faces):,}")

        if len(tri_info) > 0:
            print(f"   三角形信息: {len(tri_info)} 条记录")
            if 'type' in tri_info.columns:
                type_counts = tri_info['type'].value_counts()
                for mesh_type, count in type_counts.items():
                    print(f"     - {mesh_type}: {count:,}")
    except Exception as e:
        raise RuntimeError(f"mesh生成失败: {e}")

    # 步骤4: 保存trimesh
    print(f"\n💾 保存trimesh到: {output_mesh_path}")
    output_path = Path(output_mesh_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_trimesh.export(str(output_path))

    # 输出统计信息
    print("\n" + "="*60)
    print("📊 Mesh统计信息")
    print("="*60)
    print(f"顶点数: {len(building_trimesh.vertices):,}")
    print(f"三角形数: {len(building_trimesh.faces):,}")
    print(f"边界范围 (X): [{building_trimesh.bounds[0][0]:.2f}, {building_trimesh.bounds[1][0]:.2f}]")
    print(f"边界范围 (Y): [{building_trimesh.bounds[0][1]:.2f}, {building_trimesh.bounds[1][1]:.2f}]")
    print(f"边界范围 (Z): [{building_trimesh.bounds[0][2]:.2f}, {building_trimesh.bounds[1][2]:.2f}]")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")
    print("="*60)

    return building_trimesh


if __name__ == "__main__":
    """
    从建筑footprint转换建筑mesh

    使用方法:
    1. 在文件顶部的CONFIG区域修改参数
    2. 运行: python prepare_building_mesh_from_footprint.py

    数据要求:
    - 输入文件必须包含以下字段:
      - geometry: Polygon几何体
      - height: 建筑高度(米)
    - 推荐坐标系: WGS84 (EPSG:4326)
    """

    try:
        mesh = prepare_building_mesh_from_footprint(
            footprint_path=CONFIG['FOOTPRINT_PATH'],
            output_mesh_path=CONFIG['OUTPUT_MESH_PATH'],
            grid_size=CONFIG['GRID_SIZE'],
            n_jobs=CONFIG['N_JOBS'],
            batch_size=CONFIG['BATCH_SIZE']
        )
        print("\n✅ 转换完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
