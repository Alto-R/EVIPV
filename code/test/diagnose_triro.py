"""详细诊断 triro OptiX 初始化问题"""
import sys
import torch
import trimesh
import numpy as np

print("="*60)
print("🔍 详细诊断 triro 和 OptiX 问题")
print("="*60)

# 1. 检查 CUDA
print("\n1. CUDA 状态:")
print(f"   PyTorch 版本: {torch.__version__}")
print(f"   CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   ⚠️ CUDA 不可用 - triro 需要 CUDA!")
    sys.exit(1)

# 2. 检查 triro
print("\n2. triro 库:")
try:
    import triro
    print(f"   ✓ triro 已安装")
    print(f"   路径: {triro.__file__}")
    if hasattr(triro, '__version__'):
        print(f"   版本: {triro.__version__}")
except ImportError as e:
    print(f"   ✗ triro 未安装: {e}")
    print("\n   安装方法:")
    print("   pip install git+https://github.com/lcp29/trimesh-ray-optix")
    sys.exit(1)

# 3. 检查 OptiX 环境变量
print("\n3. OptiX 环境:")
import os
optix_paths = ['OPTIX_ROOT', 'OptiX_INSTALL_DIR', 'LD_LIBRARY_PATH']
for var in optix_paths:
    val = os.environ.get(var, None)
    if val:
        print(f"   {var}: {val}")
    else:
        print(f"   {var}: (未设置)")

# 4. 尝试导入 OptiX 组件
print("\n4. 导入 triro OptiX 组件:")
try:
    from triro.ray.ray_optix import RayMeshIntersector
    print(f"   ✓ 成功导入 RayMeshIntersector")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试创建简单 intersector
print("\n5. 测试创建 OptiX intersector:")
test_mesh = trimesh.creation.box()
print(f"   测试网格: vertices={len(test_mesh.vertices)}, faces={len(test_mesh.faces)}")

try:
    print("   正在初始化 RayMeshIntersector...")
    intersector = RayMeshIntersector(mesh=test_mesh)
    print(f"   ✓ OptiX intersector 创建成功")
    print(f"   Intersector 类型: {type(intersector)}")

    # 检查内部属性
    if hasattr(intersector, '_inner'):
        print(f"   ✓ _inner 属性存在")
    else:
        print(f"   ⚠️ _inner 属性不存在 (可能导致析构错误)")

except Exception as e:
    print(f"   ✗ OptiX intersector 创建失败:")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    print("\n完整错误栈:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试光线追踪
print("\n6. 测试光线追踪:")
try:
    # 创建从上往下的光线
    ray_origins = torch.tensor([[0, 0, 5]], dtype=torch.float32).cuda()
    ray_directions = torch.tensor([[0, 0, -1]], dtype=torch.float32).cuda()

    print(f"   光线起点: {ray_origins}")
    print(f"   光线方向: {ray_directions}")
    print("   正在执行光线追踪...")

    hit, front, ray_idx, tri_idx, location, uv = intersector.intersects_closest(
        ray_origins, ray_directions, stream_compaction=True
    )

    print(f"   ✓ 光线追踪成功!")
    print(f"   命中: {hit.cpu().numpy()}")
    print(f"   交点: {location.cpu().numpy()}")
    print(f"   射线索引: {ray_idx.cpu().numpy()}")
    print(f"   三角形索引: {tri_idx.cpu().numpy()}")

except Exception as e:
    print(f"   ✗ 光线追踪失败:")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    print("\n完整错误栈:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 测试批量光线
print("\n7. 测试批量光线追踪:")
try:
    n_rays = 1000
    ray_origins = torch.rand(n_rays, 3, dtype=torch.float32).cuda() * 4 - 2
    ray_origins[:, 2] = 5  # 所有光线从 z=5 发射
    ray_directions = torch.zeros(n_rays, 3, dtype=torch.float32).cuda()
    ray_directions[:, 2] = -1  # 向下

    print(f"   光线数量: {n_rays}")
    print("   正在执行批量追踪...")

    hit, front, ray_idx, tri_idx, location, uv = intersector.intersects_closest(
        ray_origins, ray_directions, stream_compaction=True
    )

    n_hits = len(ray_idx)
    print(f"   ✓ 批量追踪成功!")
    print(f"   命中数量: {n_hits}/{n_rays} ({n_hits/n_rays*100:.1f}%)")

except Exception as e:
    print(f"   ✗ 批量追踪失败:")
    print(f"   错误: {e}")

print("\n" + "="*60)
print("✅ 诊断完成 - triro OptiX 工作正常!")
print("="*60)
