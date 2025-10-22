# 批量处理车顶光伏发电量计算系统 (GPU加速版)

基于3D建筑模型和GPS轨迹的移动光伏板发电量批量计算工具，支持GPU加速和高精度阴影分析。

## 🌟 主要特性

- ✅ **批量处理**: 自动处理多个车辆轨迹，单GPU串行高效计算
- ✅ **自动预处理**: 处理无表头CSV，添加列名，解析datetime
- ✅ **智能气象数据**: 根据每辆车日期自动获取气象数据（智能缓存）
- ✅ **GPU加速**: 使用PyTorch + triro/OptiX加速，提速8-10倍
- ✅ **高精度阴影**: GPU光线追踪实现高精度建筑阴影分析
- ✅ **简化输出**: 扁平文件结构，全英文命名
- ✅ **建筑底面轮廓**: 使用RealSceneDL将建筑底面数据转换为3D mesh

## 📋 系统要求

### 必需依赖
  
```bash
# Python环境
Python >= 3.8

# 核心库
trimesh
pyvista
pandas
numpy
geopandas
shapely
pvlib-python
pyproj
pyyaml
tqdm

# RealSceneDL库 (用于底面轮廓到mesh的转换)
# 需要安装或添加到Python路径
# pip install -e /path/to/RealSceneDL
```

### 可选依赖 (GPU加速，强烈推荐)

```bash
# PyTorch (CUDA支持)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

详细GPU环境配置请参考：[GPU环境配置指南.md](GPU环境配置指南.md)

## 🚀 快速开始

### 步骤1：预处理轨迹数据

将原始轨迹CSV文件放入 `traj/` 目录，然后运行：

```bash
cd code
python preprocess_trajectories.py
```

**功能**：
- 自动处理 `traj/` 目录下所有CSV文件
- 添加标准列名（datetime, vehicle_id, lng, lat, speed, angle, operation_status）
- 解析datetime格式（`20190228235236` → `2019-02-28 23:52:36`）
- 数据验证（坐标范围、角度检查、去重）
- 提取车辆ID（`粤B7J7Z8` → `Z8`）

**输出**：
```
traj/
├── Z8.csv                    # 原始文件（保留）
├── 0G.csv                    # 原始文件（保留）
├── Z8_processed.csv          # ✅ 预处理后
└── 0G_processed.csv          # ✅ 预处理后
```

### 步骤2：配置参数

编辑 `batch_process_trajectories.py` 文件顶部的 `CONFIG` 字典：

```python
CONFIG = {
    'location': {
        'name': '深圳市',
        'lat': 22.543099,
        'lon': 114.057868,
    },
    'data_sources': {
        'footprint_path': 'data/shenzhen_buildings.geojson',
        'trajectory_dir': 'traj',
    },
    'pv_system': {
        'panel_area': 2.0,          # 光伏板面积(m²)
        'panel_efficiency': 0.22,   # 效率 22%
        'tilt': 5,                  # 倾角(度)
        'vehicle_height': 1.5,      # 车顶高度(m)
    },
    'computation': {
        'time_resolution_minutes': 1,  # 时间分辨率
        'use_gpu': True,               # 启用GPU
        'batch_size': 100,             # 批处理大小
        'mesh_grid_size': None,        # mesh网格大小(m), None=不细分
    },
    'output': {
        'mesh_path': 'building_mesh.ply',
        'output_dir': 'output',
    },
}
```

### 步骤3：批量计算发电量

```bash
python batch_process_trajectories.py
```

**功能**：
- 自动发现所有 `*_processed.csv` 文件
- 加载建筑mesh（一次性，所有车辆共用）
- 初始化GPU计算器（一次性）
- 串行处理每个车辆：
  1. 根据轨迹日期自动获取气象数据（缓存/API）
  2. GPU计算发电量（内部GPU高度并行）
  3. 保存独立结果

**输出**：
```
output/
├── Z8_pv_generation.csv      # 车辆Z8详细结果
├── Z8_summary.txt             # 车辆Z8统计摘要
├── 0G_pv_generation.csv      # 车辆0G详细结果
├── 0G_summary.txt             # 车辆0G统计摘要
└── batch_summary.txt          # 所有车辆汇总报告
```

## 📁 项目结构

```
code/
├── batch_process_trajectories.py   # 批量处理主脚本
├── preprocess_trajectories.py      # 轨迹预处理脚本
├── prepare_building_mesh_from_footprint.py  # 底面轮廓 → mesh转换
├── fetch_irradiance_data.py         # 太阳辐射数据获取
├── pv_calculator_gpu.py             # GPU加速计算器
├── pv_generation_pvlib.py           # 基础计算器(CPU)
├── BATCH_PROCESSING_README.md       # 批量处理详细文档
└── GPU环境配置指南.md              # GPU环境配置指南

data/
└── shenzhen_buildings.geojson       # 建筑底面轮廓数据

traj/
├── Z8.csv                           # 原始轨迹
├── 0G.csv                           # 原始轨迹
├── Z8_processed.csv                 # 预处理后 ✅
└── 0G_processed.csv                 # 预处理后 ✅

output/
├── Z8_pv_generation.csv             # 车辆Z8详细结果
├── Z8_summary.txt                   # 车辆Z8统计摘要
├── 0G_pv_generation.csv             # 车辆0G详细结果
├── 0G_summary.txt                   # 车辆0G统计摘要
└── batch_summary.txt                # 批处理汇总

openmeteo_cache/                     # 气象数据自动缓存
irradiance_data/                     # 气象数据CSV备份（可选）
building_mesh.ply                    # 建筑3D模型
```

## 🔧 高级用法

### 使用外部配置文件（可选）

如果不想修改脚本，可以创建外部 `config.yaml`：

```bash
# 创建config.yaml
cat > config.yaml << EOF
location:
  name: 深圳市
  lat: 22.543099
  lon: 114.057868

data_sources:
  footprint_path: data/shenzhen_buildings.geojson
  trajectory_dir: traj

pv_system:
  panel_area: 2.0
  panel_efficiency: 0.22
  tilt: 5
  vehicle_height: 1.5

computation:
  time_resolution_minutes: 1
  use_gpu: true
  batch_size: 100

output:
  mesh_path: building_mesh.ply
  output_dir: output
EOF

# 使用外部配置
python batch_process_trajectories.py --config config.yaml
```

### 处理单个轨迹文件

```bash
# 预处理单个文件
python preprocess_trajectories.py --input traj/Z8.csv
```

### 手动准备建筑mesh

如果 `building_mesh.ply` 不存在，批处理脚本会自动从建筑底面轮廓转换。您也可以手动准备：

```bash
python prepare_building_mesh_from_footprint.py \
    -i ../data/shenzhen_buildings.geojson \
    -o ../building_mesh.ply \
    --grid-size 10  # 可选：10米网格细分
```

**底面轮廓数据要求**：
- **geometry**: 多边形几何（Polygon）
- **height**: 建筑高度(米)
- **坐标系**: EPSG:4326 (WGS84)

GeoJSON格式示例：
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]
      },
      "properties": {
        "height": 25.0
      }
    }
  ]
}
```

## 📊 输出结果说明

### 详细结果 CSV (`{vehicle_id}_pv_generation.csv`)

| 列名 | 说明 | 单位 |
|------|------|------|
| `datetime` | 时间戳 | - |
| `lng`, `lat` | GPS坐标 | 度 |
| `angle` | 车辆朝向 | 度 |
| `is_shaded` | 是否遮挡 | 1=遮挡, 0=无遮挡 |
| `poa_global` | 平面辐照度 | W/m² |
| `cell_temp` | 电池温度 | °C |
| `ac_power` | 交流功率 | W |
| `energy_kwh` | 发电量 | kWh |

### 统计摘要 TXT (`{vehicle_id}_summary.txt`)

```
Vehicle PV Generation Summary - Z8
==================================================

Overall Statistics:
  Total Energy: 3.45 kWh
  Average Power: 145.23 W
  Peak Power: 432.10 W
  Shaded Ratio: 23.5%
  Average Cell Temperature: 38.2°C
  Calculation Time: 18.3 seconds

Hourly Generation (kWh):
  08:00 - 0.123 kWh (Avg Power: 123.4 W)
  09:00 - 0.287 kWh (Avg Power: 287.2 W)
  ...
```

### 批处理汇总 (`batch_summary.txt`)

包含所有车辆的汇总统计：
- 总发电量
- 总计算时间
- 每辆车详细统计

## ☀️ 气象数据自动机制

系统会自动根据每辆车的轨迹日期获取对应的气象数据：

```
1. 读取车辆轨迹 → 推断日期范围
   Z8: 2019-02-28 23:52 ~ 2019-03-01 00:15

2. 自动调用 fetch_and_cache_irradiance_data()

3. 检查缓存 openmeteo_cache/
   ├─ 有缓存：直接读取 ⚡ (~1-3秒)
   └─ 无缓存：从API下载 → 自动缓存 (~10-30秒)

4. 返回对应日期的气象数据
```

**缓存文件示例**：
```
openmeteo_cache/
└── openmeteo_22.5431_114.0579_2019-02-28_2019-03-01_Asia-Shanghai_1min.parquet
                ↑纬度   ↑经度    ↑开始日期   ↑结束日期     ↑时区        ↑粒度
```

**智能复用**：
- ✅ 如果 Z8 和 0G 日期范围重叠 → 共用同一个缓存文件
- ✅ 只需下载一次气象数据
- ✅ 后续车辆秒级加载

## 🎯 性能参考

### 单个轨迹计算时间（NVIDIA RTX 3080）

| 车辆 | 轨迹点数 | 计算时间 | GPU利用率 |
|------|---------|---------|----------|
| Z8 | 71,300 | ~5-10分钟 | 80-95% |
| 0G | 125,531 | ~10-20分钟 | 80-95% |

### 气象数据获取时间

| 场景 | 耗时 |
|------|------|
| 首次下载（1天，1分钟粒度） | ~10-30秒 |
| 读取缓存 | ~1-3秒 ⚡ |

### 批量处理总时间（2辆车）

- **首次运行**：~20-40分钟（包含气象数据下载）
- **后续运行**：~15-30分钟（直接读缓存）

### 性能对比（CPU vs GPU）

| 计算模式 | 1000个轨迹点 | 10000个轨迹点 | 提速比 |
|---------|------------|--------------|-------|
| CPU | ~3分钟 | ~30分钟 | 1x |
| GPU | ~20秒 | ~3分钟 | **9x** |

## ⚙️ 高级配置

### 降低时间分辨率以提高速度

编辑 `CONFIG` 或 `config.yaml`:

```python
'computation': {
    'time_resolution_minutes': 60,  # 1小时分辨率（默认：1分钟）
}
```

### 调整批处理大小 (显存管理)

如果遇到 CUDA 内存不足：

```python
'computation': {
    'batch_size': 50,  # 默认：100
}
```

### 禁用GPU (仅CPU模式)

```python
'computation': {
    'use_gpu': False,
}
```

### 细化建筑mesh（提高阴影精度）

```python
'computation': {
    'mesh_grid_size': 10,  # 10米网格细分（默认：None）
}
```

## 🐛 常见问题

### 1. 找不到预处理文件

**问题**：`No processed trajectory files found`

**解决**：
```bash
# 先运行预处理脚本
python preprocess_trajectories.py
```

### 2. 气象数据下载慢

**问题**：首次运行气象数据下载慢

**解决**：
- 第一次运行会从 Open-Meteo API 下载数据（~10-30秒）
- 数据自动缓存到 `openmeteo_cache/` 目录
- 后续运行直接读缓存，秒级加载

### 3. CUDA内存不足

**问题**：`CUDA out of memory`

**解决**：减小批处理大小
```python
CONFIG = {
    'computation': {
        'batch_size': 50,  # 默认：100
    }
}
```

### 4. datetime解析失败

**问题**：`ValueError: time data does not match format`

**解决**：检查原始CSV的datetime格式是否为 `YYYYMMDDHHmmss`
```python
# 正确格式示例
20190228235236  # 2019-02-28 23:52:36
```

### 5. 找不到RealSceneDL模块

**解决**：安装RealSceneDL或将其添加到Python路径

```bash
# 方案1: 以可编辑模式安装
cd /path/to/RealSceneDL
pip install -e .

# 方案2: 添加到Python路径
export PYTHONPATH="/path/to/RealSceneDL/src:$PYTHONPATH"
```

### 6. 无效的底面轮廓数据

**解决**：确保您的数据包含必需字段

```python
import geopandas as gpd
gdf = gpd.read_file('buildings.geojson')
print(gdf.columns)  # 必须包含 'geometry' 和 'height'
print(gdf.crs)      # 应为 EPSG:4326 (WGS84)
```

### 7. 轨迹数据格式错误

**确保CSV包含以下列**（预处理前可无表头）:
- `datetime`: 时间戳（格式：YYYYMMDDHHmmss）
- `vehicle_id`: 车牌号
- `lng`: 经度
- `lat`: 纬度
- `speed`: 速度
- `angle`: 车辆朝向角度(度，正北为0)
- `operation_status`: 运行状态

## 🔬 技术细节

### 阴影计算

- **光线追踪**: 使用triro/OptiX GPU光线追踪进行批量阴影检测
- **GPU优化**: 使用PyTorch张量批量生成光线
- **太阳位置**: 基于pvlib的太阳位置计算

### 光伏建模

- **辐照度模型**: 各向同性天空模型（与RealSceneDL源码一致）
- **POA计算**: 手动实现向量化POA公式（直射+散射+反射）
- **温度模型**: SAPM电池温度模型
- **功率计算**: 带温度修正的效率模型

### POA (Plane of Array) 计算公式

```python
# 直射分量：POA_direct = DNI × cos(AOI)
poa_direct = DNI × (cos(zenith) × cos(tilt) +
                    sin(zenith) × sin(tilt) × cos(azimuth_sun - azimuth_surf))

# 散射分量（各向同性）：POA_diffuse = DHI × (1 + cos(tilt)) / 2
poa_diffuse = DHI × (1 + cos(tilt)) / 2

# 反射分量：POA_reflected = GHI × albedo × (1 - cos(tilt)) / 2
poa_reflected = GHI × albedo × (1 - cos(tilt)) / 2

# 总POA
POA_global = poa_direct + poa_diffuse + poa_reflected
```

### 坐标系统

- **输入**: WGS84 (GPS坐标)
- **内部**: 局部笛卡尔坐标
- **转换**: RealSceneDL坐标工具

### 批量处理策略

- **单GPU串行**: 所有轨迹在单个GPU上串行处理
- **GPU内部并行**: 每个轨迹内部GPU高度并行计算
- **资源共享**: 建筑mesh和计算器在所有轨迹间共享

## 📚 相关文档

- [GPU环境配置指南.md](GPU环境配置指南.md) - GPU环境配置详细说明
- **pvlib-python**: https://pvlib-python.readthedocs.io/
- **Open-Meteo API**: https://open-meteo.com/
- **PyVista光线追踪**: https://docs.pyvista.org/
- **RealSceneDL**: 建筑底面轮廓到3D mesh转换库

## 📧 联系方式

如有问题，请联系项目维护者。

## 📄 许可证

本项目仅供研究使用。
