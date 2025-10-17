"""
从3D Tiles数据转换为建筑mesh（方案A）
用于车顶光伏阴影计算
"""

import trimesh
import pyvista as pv
import os
import sys
from pathlib import Path

# 添加RealSceneDL库路径
REALSCENEDL_PATH = r"D:\1-PKU\PKU\1 Master\Projects\RealSceneDL\src"
if REALSCENEDL_PATH not in sys.path:
    sys.path.insert(0, REALSCENEDL_PATH)

from RealSceneDL.utils.merge_scene import merge_scene


def prepare_building_mesh_from_3dtiles(tileset_path, output_mesh_path, simplify=False, target_faces=None):
    """
    从3D Tiles数据转换为PyVista mesh用于光伏计算

    Parameters
    ----------
    tileset_path : str
        3D Tiles的tileset.json路径
    output_mesh_path : str
        输出mesh文件路径(.vtk格式)
    simplify : bool, optional
        是否简化mesh以减少计算量, 默认False
    target_faces : int, optional
        简化后的目标三角形数量, 仅在simplify=True时有效

    Returns
    -------
    pyvista.PolyData
        建筑物mesh对象
    """
    print("="*60)
    print("🏗️  从3D Tiles转换建筑mesh")
    print("="*60)

    tileset_path = Path(tileset_path)
    if not tileset_path.exists():
        raise FileNotFoundError(f"Tileset文件不存在: {tileset_path}")

    print(f"\n📂 输入: {tileset_path}")
    print(f"📂 输出: {output_mesh_path}")

    # 方法1: 直接加载tileset.json（推荐）
    print("\n🔄 正在加载3D Tiles场景...")
    try:
        scene = trimesh.load(str(tileset_path), force='scene')
        print(f"✅ 成功加载场景，包含 {len(scene.geometry)} 个模型")

        # 显示场景信息
        total_vertices = 0
        total_faces = 0
        for name, geom in scene.geometry.items():
            if hasattr(geom, 'vertices'):
                total_vertices += len(geom.vertices)
            if hasattr(geom, 'faces'):
                total_faces += len(geom.faces)

        print(f"   总顶点数: {total_vertices:,}")
        print(f"   总三角形数: {total_faces:,}")

    except Exception as e:
        print(f"❌ 直接加载失败: {e}")
        print("🔄 尝试逐个转换GLB文件...")

        # 方法2: 逐个处理GLB文件（备用方案）
        tileset_dir = tileset_path.parent
        glb_files = list(tileset_dir.glob('*.glb'))
        print(f"   找到 {len(glb_files)} 个GLB文件")

        if not glb_files:
            raise RuntimeError("未找到GLB文件，请检查3D Tiles数据")

        meshes = []
        for i, glb_file in enumerate(glb_files):
            try:
                mesh = trimesh.load(str(glb_file))
                meshes.append(mesh)
                if (i + 1) % 10 == 0:
                    print(f"   已加载 {i+1}/{len(glb_files)} 个文件")
            except Exception as e:
                print(f"   ⚠️  跳过文件 {glb_file.name}: {e}")
                continue

        if not meshes:
            raise RuntimeError("未能成功加载任何GLB文件")

        scene = trimesh.Scene(meshes)
        print(f"✅ 成功加载 {len(meshes)} 个模型")

    # 合并为单个trimesh
    print("\n🔗 合并所有模型为单个mesh...")
    building_trimesh = merge_scene(scene)

    print(f"   合并后顶点数: {len(building_trimesh.vertices):,}")
    print(f"   合并后三角形数: {len(building_trimesh.faces):,}")

    # 可选: 简化mesh
    if simplify and target_faces is not None:
        print(f"\n⚡ 简化mesh到目标三角形数: {target_faces:,}")
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(building_trimesh.vertices, building_trimesh.faces))
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

    # 转换为PyVista格式（用于光线追踪）
    print("\n🔄 转换为PyVista格式...")
    building_mesh_pv = pv.wrap(building_trimesh)

    # 保存mesh
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

    parser = argparse.ArgumentParser(description='从3D Tiles转换建筑mesh')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='3D Tiles的tileset.json路径')
    parser.add_argument('-o', '--output', type=str, default='building_mesh.vtk',
                        help='输出mesh文件路径(.vtk)')
    parser.add_argument('--simplify', action='store_true',
                        help='是否简化mesh')
    parser.add_argument('--target-faces', type=int, default=1000000,
                        help='简化后的目标三角形数量')

    args = parser.parse_args()

    try:
        mesh = prepare_building_mesh_from_3dtiles(
            tileset_path=args.input,
            output_mesh_path=args.output,
            simplify=args.simplify,
            target_faces=args.target_faces
        )
        print("\n✅ 转换完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
