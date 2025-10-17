# 车顶光伏发电量计算系统 (GPU加速版)

**中文** | [English](README.md)

基于3D建筑模型和GPS轨迹的移动光伏板发电量计算工具，支持GPU加速和高精度阴影分析。

## 🌟 主要特性

- ✅ **直接3D Tiles集成**: 从Google 3D Tiles数据直接生成建筑mesh
- ✅ **智能缓存**: 太阳辐射数据自动缓存，避免重复下载
- ✅ **GPU加速**: 使用PyTorch加速批量计算，提速8-10倍
- ✅ **高精度**: 支持1分钟时间分辨率
- ✅ **完整流程**: 从数据准备到结果分析的一站式解决方案

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
pvlib-python
pyproj
pyyaml
tqdm

# RealSceneDL库
# 需要添加到Python路径
```

### 可选依赖 (GPU加速)

```bash
# PyTorch (CUDA支持)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 快速开始

### 1. 生成默认配置文件

```bash
cd code
python main_pv_calculation_gpu.py --create-config
```

这将创建 `config.yaml` 配置文件。

### 2. 修改配置文件

编辑 `config.yaml`，设置您的参数：

```yaml
location:
  name: 深圳市
  lat: 22.543099
  lon: 114.057868

data_sources:
  3d_tiles_path: data/shenzhen_3dtiles/tileset.json
  trajectory_path: traj/onetra_0312_1.csv

pv_system:
  panel_area: 2.0          # 光伏板面积(m²)
  panel_efficiency: 0.22   # 效率 22%
  tilt: 5                  # 倾角(度)
  vehicle_height: 1.5      # 车顶高度(m)

computation:
  time_resolution_minutes: 1  # 1分钟分辨率
  use_gpu: true              # 启用GPU
  batch_size: 100

output:
  mesh_path: building_mesh.vtk
  result_path: output/pv_generation_1min_gpu.csv
```

### 3. 运行完整流程

```bash
# 使用配置文件
python main_pv_calculation_gpu.py --config config.yaml

# 或使用命令行参数
python main_pv_calculation_gpu.py \
    --lat 22.543099 \
    --lon 114.057868 \
    --date 2019-03-12 \
    --tileset data/shenzhen_3dtiles/tileset.json \
    --trajectory traj/onetra_0312_1.csv
```

## 📁 项目结构

```
.
├── code/                                   # 源代码
│   ├── main_pv_calculation_gpu.py          # 主执行脚本
│   ├── prepare_building_mesh_from_3dtiles.py  # 3D Tiles → mesh转换
│   ├── fetch_irradiance_data.py            # 太阳辐射数据获取
│   ├── pv_calculator_gpu.py                # GPU加速计算器
│   ├── pv_generation_pvlib.py              # 基础计算器(CPU)
│   └── config.yaml                         # 配置文件
│
├── traj/                                   # GPS轨迹数据
│   └── onetra_0312_1.csv
│
├── output/                                 # 计算结果
│   ├── pv_generation_1min_gpu.csv
│   └── pv_generation_1min_gpu_summary.txt
│
├── irradiance_data/                        # 辐射数据CSV备份
├── openmeteo_cache/                        # 自动缓存(Parquet格式)
├── building_mesh.vtk                       # 建筑mesh文件
│
└── README_CN.md                            # 本文件
```

## 🔧 分步执行

如果您想分步运行，可以单独执行各个模块：

### 步骤1: 转换建筑mesh

```bash
cd code
python prepare_building_mesh_from_3dtiles.py \
    -i ../data/shenzhen_3dtiles/tileset.json \
    -o ../building_mesh.vtk
```

**可选的mesh简化**:
```bash
python prepare_building_mesh_from_3dtiles.py \
    -i ../data/shenzhen_3dtiles/tileset.json \
    -o ../building_mesh.vtk \
    --simplify \
    --target-faces 1000000
```

### 步骤2: 获取辐射数据

```bash
cd code
python fetch_irradiance_data.py \
    --lat 22.543099 \
    --lon 114.057868 \
    --start 2019-03-12 \
    --end 2019-03-12 \
    --granularity 1min
```

**主要功能**:
- 自动缓存到 `openmeteo_cache/` (Parquet格式)
- 可选CSV导出到 `irradiance_data/`
- 数据质量报告

### 步骤3: 运行计算

使用主脚本配合预处理的数据可以更快执行。

## 📊 输出结果

### 主要输出文件

1. **pv_generation_1min_gpu.csv** - 详细结果
   - `datetime`: 时间戳
   - `lng`, `lat`: GPS坐标
   - `angle`: 车辆朝向
   - `is_shaded`: 是否被遮挡 (1=遮挡, 0=无遮挡)
   - `poa_global`: 平面辐照度 (W/m²)
   - `cell_temp`: 电池温度 (°C)
   - `ac_power`: 交流功率 (W)
   - `energy_kwh`: 发电量 (kWh)

2. **pv_generation_1min_gpu_summary.txt** - 统计摘要
   - 总发电量
   - 平均/峰值功率
   - 遮阴比例
   - 逐小时统计

### 结果示例

```
总体统计:
  总发电量: 3.45 kWh
  平均功率: 145.23 W
  峰值功率: 432.10 W
  遮阴时长占比: 23.5%
  平均电池温度: 38.2°C
  计算耗时: 18.3 秒

逐小时发电量 (kWh):
  08:00 - 0.123 kWh (平均功率: 123.4 W)
  09:00 - 0.287 kWh (平均功率: 287.2 W)
  ...
```

## 🎯 性能对比

| 计算模式 | 1000个轨迹点 | 10000个轨迹点 | 提速比 |
|---------|------------|--------------|-------|
| CPU | ~3分钟 | ~30分钟 | 1x |
| GPU (本脚本) | ~20秒 | ~3分钟 | **9x** |

*测试环境: NVIDIA RTX 3080, Intel i7-12700K*

## ⚙️ 高级选项

### 降低时间分辨率以提高速度

修改 `config.yaml`:

```yaml
computation:
  time_resolution_minutes: 60  # 1小时分辨率
```

### 禁用GPU (仅CPU模式)

```bash
cd code
python main_pv_calculation_gpu.py --config config.yaml --no-gpu
```

### 调整批处理大小 (显存管理)

```yaml
computation:
  batch_size: 50  # 默认: 100, 如果CUDA内存不足可减小
```

## 🐛 常见问题

### 1. CUDA out of memory

**解决**: 在 `config.yaml` 中减小批处理大小

```yaml
computation:
  batch_size: 50  # 默认100
```

### 2. 找不到RealSceneDL模块

**解决**: 检查 `code/` 目录下各脚本中的路径设置

```python
REALSCENEDL_PATH = r"D:\1-PKU\PKU\1 Master\Projects\RealSceneDL\src"
```

根据您的系统更新此路径。

### 3. 辐射数据下载慢

**解决**:
- 第一次下载后会自动缓存到 `openmeteo_cache/`
- 后续运行会直接读取缓存，非常快
- 1小时数据会自动插值到1分钟分辨率

### 4. 轨迹数据格式错误

**确保CSV包含以下列**:
- `datetime`: 时间戳 (可解析为datetime)
- `lng`: 经度
- `lat`: 纬度
- `angle`: 车辆朝向角度(度，正北为0)

## 🔬 技术细节

### 阴影计算

- **光线追踪**: 使用PyVista多光线追踪进行批量阴影检测
- **GPU优化**: 使用PyTorch张量批量生成光线
- **太阳位置**: 基于pvlib的太阳位置计算

### 光伏建模

- **辐照度模型**: 各向同性天空模型 (pvlib)
- **温度模型**: SAPM电池温度模型
- **功率计算**: 带温度修正的简单效率模型

### 坐标系统

- **输入**: WGS84 (GPS坐标)
- **内部**: 局部笛卡尔坐标
- **转换**: RealSceneDL坐标工具

## 📚 参考文献

- **pvlib-python**: https://pvlib-python.readthedocs.io/
- **Open-Meteo API**: https://open-meteo.com/
- **PyVista光线追踪**: https://docs.pyvista.org/
- **Google 3D Tiles**: https://developers.google.com/maps/documentation/tile

## 📄 许可证

本项目仅供研究使用。

## 📧 联系方式

如有问题，请联系项目维护者。
