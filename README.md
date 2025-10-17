# Vehicle Rooftop PV Generation Calculator (GPU-Accelerated)

[中文文档](README_CN.md) | **English**

A GPU-accelerated tool for calculating photovoltaic power generation on moving vehicles using 3D building models and GPS trajectories, with support for high-resolution shadow analysis.

## 🌟 Key Features

- ✅ **Direct 3D Tiles Integration**: Convert Google 3D Tiles data to building meshes
- ✅ **Intelligent Caching**: Automatic solar irradiance data caching to avoid redundant downloads
- ✅ **GPU Acceleration**: PyTorch-powered batch computation with 8-10x speedup
- ✅ **High Precision**: Support for 1-minute time resolution
- ✅ **End-to-End Pipeline**: Complete workflow from data preparation to result analysis

## 📋 System Requirements

### Required Dependencies

```bash
# Python Environment
Python >= 3.11

# Core Libraries
trimesh
pyvista
pandas
numpy
geopandas
pvlib-python
pyproj
pyyaml
tqdm

# RealSceneDL Library
# Must be added to Python path
```

### Optional Dependencies (GPU Acceleration)

```bash
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Quick Start

### 1. Generate Default Configuration

```bash
cd code
python main_pv_calculation_gpu.py --create-config
```

This creates a `config.yaml` configuration file.

### 2. Edit Configuration

Modify `config.yaml` with your parameters:

```yaml
location:
  name: Shenzhen
  lat: 22.543099
  lon: 114.057868

data_sources:
  3d_tiles_path: data/shenzhen_3dtiles/tileset.json
  trajectory_path: traj/onetra_0312_1.csv

pv_system:
  panel_area: 2.0          # PV panel area (m²)
  panel_efficiency: 0.22   # Efficiency 22%
  tilt: 5                  # Tilt angle (degrees)
  vehicle_height: 1.5      # Rooftop height (m)

computation:
  time_resolution_minutes: 1  # 1-minute resolution
  use_gpu: true              # Enable GPU
  batch_size: 100

output:
  mesh_path: building_mesh.vtk
  result_path: output/pv_generation_1min_gpu.csv
```

### 3. Run Complete Pipeline

```bash
# Using configuration file
python main_pv_calculation_gpu.py --config config.yaml

# Or using command-line arguments
python main_pv_calculation_gpu.py \
    --lat 22.543099 \
    --lon 114.057868 \
    --date 2019-03-12 \
    --tileset data/shenzhen_3dtiles/tileset.json \
    --trajectory traj/onetra_0312_1.csv
```

## 📁 Project Structure

```
.
├── code/                                   # Source code
│   ├── main_pv_calculation_gpu.py          # Main execution script
│   ├── prepare_building_mesh_from_3dtiles.py  # 3D Tiles → mesh conversion
│   ├── fetch_irradiance_data.py            # Solar irradiance data fetching
│   ├── pv_calculator_gpu.py                # GPU-accelerated calculator
│   ├── pv_generation_pvlib.py              # Base calculator (CPU)
│   └── config.yaml                         # Configuration file
│
├── traj/                                   # GPS trajectory data
│   └── onetra_0312_1.csv
│
├── output/                                 # Calculation results
│   ├── pv_generation_1min_gpu.csv
│   └── pv_generation_1min_gpu_summary.txt
│
├── irradiance_data/                        # Irradiance data CSV backup
├── openmeteo_cache/                        # Auto-cache (Parquet format)
├── building_mesh.vtk                       # Building mesh file
│
└── README.md                               # This file
```

## 🔧 Step-by-Step Execution

If you prefer to run each step individually:

### Step 1: Convert Building Mesh

```bash
cd code
python prepare_building_mesh_from_3dtiles.py \
    -i ../data/shenzhen_3dtiles/tileset.json \
    -o ../building_mesh.vtk
```

**Optional mesh simplification**:
```bash
python prepare_building_mesh_from_3dtiles.py \
    -i ../data/shenzhen_3dtiles/tileset.json \
    -o ../building_mesh.vtk \
    --simplify \
    --target-faces 1000000
```

### Step 2: Fetch Irradiance Data

```bash
cd code
python fetch_irradiance_data.py \
    --lat 22.543099 \
    --lon 114.057868 \
    --start 2019-03-12 \
    --end 2019-03-12 \
    --granularity 1min
```

**Key features**:
- Automatic caching to `openmeteo_cache/` (Parquet format)
- Optional CSV export to `irradiance_data/`
- Data quality reporting

### Step 3: Run PV Calculation

Use the main script with pre-processed data for faster execution.

## 📊 Output Results

### Main Output Files

1. **pv_generation_1min_gpu.csv** - Detailed results
   - `datetime`: Timestamp
   - `lng`, `lat`: GPS coordinates
   - `angle`: Vehicle orientation
   - `is_shaded`: Shadow status (1=shaded, 0=unshaded)
   - `poa_global`: Plane of array irradiance (W/m²)
   - `cell_temp`: Cell temperature (°C)
   - `ac_power`: AC power output (W)
   - `energy_kwh`: Energy generation (kWh)

2. **pv_generation_1min_gpu_summary.txt** - Statistical summary
   - Total energy generation
   - Average/peak power
   - Shading ratio
   - Hourly statistics

### Example Output

```
Overall Statistics:
  Total Energy: 3.45 kWh
  Average Power: 145.23 W
  Peak Power: 432.10 W
  Shading Ratio: 23.5%
  Average Cell Temperature: 38.2°C
  Computation Time: 18.3 seconds

Hourly Energy Generation (kWh):
  08:00 - 0.123 kWh (Avg Power: 123.4 W)
  09:00 - 0.287 kWh (Avg Power: 287.2 W)
  ...
```

## 🎯 Performance Comparison

| Mode | 1000 Points | 10000 Points | Speedup |
|------|------------|--------------|---------|
| CPU | ~3 min | ~30 min | 1x |
| GPU (This Script) | ~20 sec | ~3 min | **9x** |

*Test Environment: NVIDIA RTX 3080, Intel i7-12700K*

## ⚙️ Advanced Options

### Reduce Time Resolution for Speed

Modify `config.yaml`:

```yaml
computation:
  time_resolution_minutes: 60  # 1-hour resolution
```

### Disable GPU (CPU-only Mode)

```bash
cd code
python main_pv_calculation_gpu.py --config config.yaml --no-gpu
```

### Adjust Batch Size (Memory Management)

```yaml
computation:
  batch_size: 50  # Default: 100, reduce if CUDA out of memory
```

## 🐛 Troubleshooting

### 1. CUDA Out of Memory

**Solution**: Reduce batch size in `config.yaml`

```yaml
computation:
  batch_size: 50  # Default: 100
```

### 2. RealSceneDL Module Not Found

**Solution**: Verify the path in each script under `code/`

```python
REALSCENEDL_PATH = r"D:\1-PKU\PKU\1 Master\Projects\RealSceneDL\src"
```

Update the path to match your system.

### 3. Slow Irradiance Data Download

**Solution**:
- First download automatically cached to `openmeteo_cache/`
- Subsequent runs read from cache (very fast)
- Hourly data automatically interpolated to 1-minute

### 4. Trajectory Data Format Error

**Ensure CSV contains**:
- `datetime`: Timestamp (parseable to datetime)
- `lng`: Longitude
- `lat`: Latitude
- `angle`: Vehicle heading angle (degrees, 0=North)

## 🔬 Technical Details

### Shadow Calculation

- **Ray Tracing**: PyVista multi-ray tracing for batch shadow detection
- **GPU Optimization**: Batch generation of rays using PyTorch tensors
- **Sun Position**: pvlib solar position calculations

### PV Modeling

- **Irradiance Model**: Isotropic sky model (pvlib)
- **Temperature Model**: SAPM cell temperature model
- **Power Calculation**: Simple efficiency model with temperature correction

### Coordinate Systems

- **Input**: WGS84 (GPS coordinates)
- **Internal**: Local Cartesian coordinates
- **Transformation**: RealSceneDL coordinate utilities

## 📚 References

- **pvlib-python**: https://pvlib-python.readthedocs.io/
- **Open-Meteo API**: https://open-meteo.com/
- **PyVista Ray Tracing**: https://docs.pyvista.org/
- **Google 3D Tiles**: https://developers.google.com/maps/documentation/tile

## 📄 License

This project is for research purposes only.

## 📧 Contact

For questions and issues, please contact the project maintainer.
