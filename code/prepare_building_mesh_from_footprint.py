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
import os
import sys
from pathlib import Path

# 导入RealSceneDL库的footprint转换功能
from RealSceneDL.footprint import footprint_to_tmesh, footprint_preprocess

def prepare_building_mesh_from_footprint(
    footprint_path,
    output_mesh_path,
    grid_size=None,
    simplify=False,
    target_faces=None
):
    """
    从建筑footprint数据转换为PyVista mesh用于光伏计算

    Parameters
    ----------
    footprint_path : str
        建筑footprint数据路径，支持GeoJSON, Shapefile, GeoPackage等格式
    output_mesh_path : str
        输出mesh文件路径(.vtk格式)
    grid_size : float, optional
        mesh网格细分精度(米)。如果指定，会对屋顶和墙面进行精细网格划分，
        适用于高精度太阳能分析。默认None(不细分)
    simplify : bool, optional
        是否简化mesh以减少计算量, 默认False
    target_faces : int, optional
        简化后的目标三角形数量, 仅在simplify=True时有效

    Returns
    -------
    pyvista.PolyData
        建筑物mesh对象

    Raises
    ------
    FileNotFoundError
        如果footprint文件不存在
    ValueError
        如果footprint数据缺少必需字段(geometry, height)

    Examples
    --------
    >>> # 基本使用
    >>> mesh = prepare_building_mesh_from_footprint(
    ...     'buildings.geojson',
    ...     'building_mesh.vtk'
    ... )

    >>> # 使用网格细分进行精细分析
    >>> mesh = prepare_building_mesh_from_footprint(
    ...     'buildings.geojson',
    ...     'building_mesh.vtk',
    ...     grid_size=10  # 10米网格
    ... )
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
        building_gdf = footprint_preprocess(building_gdf)
        print(f"✅ 预处理完成，保留 {len(building_gdf)} 个有效建筑")
    except Exception as e:
        print(f"⚠️  预处理出现问题: {e}")
        print("   尝试继续处理...")

    # 步骤3: 转换为trimesh
    print("\n🔄 转换为3D mesh...")
    try:
        building_trimesh, tri_info = footprint_to_tmesh(
            building_gdf,
            grid_size=grid_size
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

    # 可选: 简化mesh
    if simplify and target_faces is not None:
        print(f"\n⚡ 简化mesh到目标三角形数: {target_faces:,}")
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                building_trimesh.vertices,
                building_trimesh.faces
            ))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            simplified_mesh = ms.current_mesh()
            building_trimesh = trimesh.Trimesh(
                vertices=simplified_mesh.vertex_matrix(),
                faces=simplified_mesh.face_matrix()
            )
            print(f"✅ 简化完成，新三角形数: {len(building_trimesh.faces):,}")
        except ImportError:
            print("⚠️  pymeshlab未安装，跳过简化步骤")
        except Exception as e:
            print(f"⚠️  简化失败: {e}，使用原始mesh")

    # 步骤4: 转换为PyVista格式（用于光线追踪）
    print("\n🔄 转换为PyVista格式...")
    building_mesh_pv = pv.wrap(building_trimesh)

    # 步骤5: 保存mesh
    print(f"\n💾 保存mesh到: {output_mesh_path}")
    output_path = Path(output_mesh_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_mesh_pv.save(str(output_path))

    # 输出统计信息
    print("\n" + "="*60)
    print("📊 Mesh统计信息")
    print("="*60)
    print(f"顶点数: {building_mesh_pv.n_points:,}")
    print(f"三角形数: {building_mesh_pv.n_faces:,}")
    print(f"边界范围 (X): [{building_mesh_pv.bounds[0]:.2f}, {building_mesh_pv.bounds[1]:.2f}]")
    print(f"边界范围 (Y): [{building_mesh_pv.bounds[2]:.2f}, {building_mesh_pv.bounds[3]:.2f}]")
    print(f"边界范围 (Z): [{building_mesh_pv.bounds[4]:.2f}, {building_mesh_pv.bounds[5]:.2f}]")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")
    print("="*60)

    return building_mesh_pv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='从建筑footprint转换建筑mesh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本转换
  python prepare_building_mesh_from_footprint.py -i buildings.geojson -o building_mesh.vtk

  # 使用网格细分进行精细分析
  python prepare_building_mesh_from_footprint.py -i buildings.geojson -o building_mesh.vtk --grid-size 10

  # 简化mesh以减少计算量
  python prepare_building_mesh_from_footprint.py -i buildings.geojson -o building_mesh.vtk --simplify --target-faces 1000000

数据要求:
  输入文件必须包含以下字段:
  - geometry: Polygon几何体
  - height: 建筑高度(米)

  推荐坐标系: WGS84 (EPSG:4326)
        """
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='建筑footprint数据路径 (GeoJSON/Shapefile/GeoPackage)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='building_mesh.vtk',
        help='输出mesh文件路径 (.vtk)'
    )
    parser.add_argument(
        '--grid-size',
        type=float,
        default=None,
        help='mesh网格细分精度(米)，用于精细太阳能分析'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='是否简化mesh'
    )
    parser.add_argument(
        '--target-faces',
        type=int,
        default=1000000,
        help='简化后的目标三角形数量'
    )

    args = parser.parse_args()

    try:
        mesh = prepare_building_mesh_from_footprint(
            footprint_path=args.input,
            output_mesh_path=args.output,
            grid_size=args.grid_size,
            simplify=args.simplify,
            target_faces=args.target_faces
        )
        print("\n✅ 转换完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
