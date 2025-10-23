# Batch Vehicle Rooftop PV Generation Calculator (GPU-Accelerated)

[中文文档](README_CN.md) | **English**

A GPU-accelerated batch processing tool for calculating photovoltaic power generation on moving vehicles using 3D building models and GPS trajectories, with support for high-precision shadow analysis.

## 🌟 Key Features

- ✅ **Batch Processing**: Automatically process multiple vehicle trajectories with efficient single-GPU serial computation
- ✅ **Automatic Preprocessing**: Handle headerless CSV files, add column names, parse datetime
- ✅ **Intelligent Weather Data**: Automatically fetch weather data based on each vehicle's date range (smart caching)
- ✅ **GPU Acceleration**: PyTorch + triro/OptiX acceleration with 8-10x speedup
- ✅ **High-Precision Shadow**: GPU ray tracing for accurate building shadow analysis
- ✅ **Simplified Output**: Flat file structure with English-only naming
- ✅ **Building Footprint**: Convert building footprint data to 3D meshes using RealSceneDL

## 📋 System Requirements

### Required Dependencies

```bash
# Python Environment
Python >= 3.8

# Core Libraries
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

# RealSceneDL Library (for footprint to mesh conversion)
# Must be installed or added to Python path
# pip install -e /path/to/RealSceneDL
```

### Optional Dependencies (GPU Acceleration, Highly Recommended)

```bash
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

For detailed GPU environment setup, see: [code/GPU环境配置指南.md](code/GPU环境配置指南.md)

## 🚀 Quick Start

### Step 1: Preprocess Trajectory Data

Place raw trajectory CSV files in the `traj/` directory, then run:

```bash
cd code
python preprocess_trajectories.py
```

**Features**:
- Automatically process all CSV files in `traj/` directory
- Add standard column names (datetime, vehicle_id, lng, lat, speed, angle, operation_status)
- Parse datetime format (`20190228235236` → `2019-02-28 23:52:36`)
- Data validation (coordinate range, angle check, deduplication)
- Extract vehicle ID (`粤B7J7Z8` → `Z8`)

**Output**:
```
traj/
├── Z8.csv                    # Original file (retained)
├── 0G.csv                    # Original file (retained)
├── Z8_processed.csv          # ✅ Preprocessed
└── 0G_processed.csv          # ✅ Preprocessed
```

### Step 2: Configure Parameters

Edit the `CONFIG` dictionary at the top of `batch_process_trajectories.py`:

```python
CONFIG = {
    'location': {
        'name': 'Shenzhen',
        'lat': 22.543099,
        'lon': 114.057868,
    },
    'data_sources': {
        'footprint_path': 'data/shenzhen_buildings.geojson',
        'trajectory_dir': 'traj',
    },
    'pv_system': {
        'panel_area': 2.0,          # PV panel area (m²)
        'panel_efficiency': 0.22,   # Efficiency 22%
        'tilt': 5,                  # Tilt angle (degrees)
        'vehicle_height': 1.5,      # Rooftop height (m)
    },
    'computation': {
        'time_resolution_minutes': 1,  # Time resolution
        'use_gpu': True,               # Enable GPU
        'batch_size': 100,             # Batch size
        'mesh_grid_size': None,        # Mesh grid size (m), None = no subdivision
    },
    'output': {
        'mesh_path': 'building_mesh.ply',
        'output_dir': 'output',
    },
}
```

### Step 3: Batch Calculate PV Generation

```bash
python batch_process_trajectories.py
```

**Features**:
- Automatically discover all `*_processed.csv` files
- Load building mesh once (shared by all vehicles)
- Initialize GPU calculator once
- Process each vehicle serially:
  1. Automatically fetch weather data based on trajectory dates (cache/API)
  2. GPU-accelerated PV generation calculation (internal GPU parallelism)
  3. Save individual results

**Output**:
```
output/
├── Z8_pv_generation.csv      # Vehicle Z8 detailed results
├── 0G_pv_generation.csv      # Vehicle 0G detailed results
└── batch_summary.txt          # Batch processing summary
```

## 📁 Project Structure

```
.
├── code/                                   # Source code
│   ├── batch_process_trajectories.py      # Batch processing main script
│   ├── preprocess_trajectories.py         # Trajectory preprocessing script
│   ├── prepare_building_mesh_from_footprint.py  # Footprint → mesh conversion
│   ├── fetch_irradiance_data.py           # Solar irradiance data fetching
│   ├── pv_calculator_gpu.py               # GPU-accelerated calculator
│   ├── pv_generation_pvlib.py             # Base calculator (CPU)
│   ├── BATCH_PROCESSING_README.md         # Detailed batch processing guide
│   └── GPU环境配置指南.md                 # GPU environment setup guide
│
├── data/                                   # Input data
│   └── shenzhen_buildings.geojson         # Building footprint data
│
├── traj/                                   # GPS trajectory data
│   ├── Z8.csv                             # Raw trajectory
│   ├── 0G.csv                             # Raw trajectory
│   ├── Z8_processed.csv                   # Preprocessed ✅
│   └── 0G_processed.csv                   # Preprocessed ✅
│
├── output/                                 # Calculation results
│   ├── Z8_pv_generation.csv               # Vehicle Z8 detailed results
│   ├── 0G_pv_generation.csv               # Vehicle 0G detailed results
│   └── batch_summary.txt                  # Batch processing summary
│
├── openmeteo_cache/                        # Automatic weather data caching
├── irradiance_data/                        # Weather data CSV backup (optional)
├── building_mesh.ply                       # 3D building model
│
├── README.md                               # This file
└── README_CN.md                            # Chinese documentation
```

## 🔧 Advanced Usage

### Using External Config File (Optional)

If you don't want to modify the script, create an external `config.yaml`:

```bash
# Create config.yaml
cat > config.yaml << EOF
location:
  name: Shenzhen
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

# Use external config
cd code
python batch_process_trajectories.py --config config.yaml
```

### Processing Individual Trajectory Files

```bash
cd code
# Preprocess single file
python preprocess_trajectories.py --input traj/Z8.csv
```

### Manually Prepare Building Mesh

If `building_mesh.ply` doesn't exist, the batch processing script will automatically convert from building footprints. You can also prepare it manually:

```bash
cd code
python prepare_building_mesh_from_footprint.py \
    -i ../data/shenzhen_buildings.geojson \
    -o ../building_mesh.ply \
    --grid-size 10  # Optional: 10-meter grid subdivision
```

**Footprint Data Requirements**:
- **geometry**: Polygon geometry
- **height**: Building height in meters
- **CRS**: EPSG:4326 (WGS84)

Example GeoJSON format:
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

## 📊 Output Results

### Detailed Results CSV (`{vehicle_id}_pv_generation.csv`)

| Column | Description | Unit |
|--------|-------------|------|
| `datetime` | Timestamp | - |
| `lng`, `lat` | GPS coordinates | degrees |
| `angle` | Vehicle heading | degrees |
| `is_shaded` | Shadow status | 1=shaded, 0=unshaded |
| `poa_global` | Plane of array irradiance | W/m² |
| `cell_temp` | Cell temperature | °C |
| `ac_power` | AC power | W |
| `energy_kwh` | Energy generation | kWh |

### Batch Processing Summary (`batch_summary.txt`)

Contains aggregate statistics for all vehicles:
- Total energy generation
- Total computation time
- Detailed statistics per vehicle

## ☀️ Automatic Weather Data Mechanism

The system automatically fetches weather data based on each vehicle's trajectory dates:

```
1. Read vehicle trajectory → Infer date range
   Z8: 2019-02-28 23:52 ~ 2019-03-01 00:15

2. Automatically call fetch_and_cache_irradiance_data()

3. Check cache openmeteo_cache/
   ├─ Cache exists: Load directly ⚡ (~1-3 seconds)
   └─ No cache: Download from API → Auto-cache (~10-30 seconds)

4. Return weather data for corresponding dates
```

**Cache File Example**:
```
openmeteo_cache/
└── openmeteo_22.5431_114.0579_2019-02-28_2019-03-01_Asia-Shanghai_1min.parquet
                ↑latitude ↑longitude ↑start date  ↑end date     ↑timezone     ↑granularity
```

**Intelligent Reuse**:
- ✅ If Z8 and 0G date ranges overlap → Share same cache file
- ✅ Download weather data only once
- ✅ Subsequent vehicles load in seconds

## 🎯 Performance Benchmarks

### Single Trajectory Computation Time (NVIDIA RTX 3080)

| Vehicle | Trajectory Points | Computation Time | GPU Utilization |
|---------|-------------------|------------------|-----------------|
| Z8 | 71,300 | ~5-10 minutes | 80-95% |
| 0G | 125,531 | ~10-20 minutes | 80-95% |

### Weather Data Fetch Time

| Scenario | Time |
|----------|------|
| First download (1 day, 1-min granularity) | ~10-30 seconds |
| Load from cache | ~1-3 seconds ⚡ |

### Batch Processing Total Time (2 vehicles)

- **First run**: ~20-40 minutes (includes weather data download)
- **Subsequent runs**: ~15-30 minutes (direct cache loading)

### Performance Comparison (CPU vs GPU)

| Mode | 1000 Points | 10000 Points | Speedup |
|------|-------------|--------------|---------|
| CPU | ~3 min | ~30 min | 1x |
| GPU | ~20 sec | ~3 min | **9x** |

## ⚙️ Advanced Configuration

### Reduce Time Resolution for Speed

Edit `CONFIG` or `config.yaml`:

```python
'computation': {
    'time_resolution_minutes': 60,  # 1-hour resolution (default: 1 minute)
}
```

### Adjust Batch Size (Memory Management)

If encountering CUDA out of memory:

```python
'computation': {
    'batch_size': 50,  # Default: 100
}
```

### Disable GPU (CPU-only Mode)

```python
'computation': {
    'use_gpu': False,
}
```

### Refine Building Mesh (Improve Shadow Accuracy)

```python
'computation': {
    'mesh_grid_size': 10,  # 10-meter grid subdivision (default: None)
}
```

## 🐛 Troubleshooting

### 1. Preprocessed Files Not Found

**Problem**: `No processed trajectory files found`

**Solution**:
```bash
# Run preprocessing script first
cd code
python preprocess_trajectories.py
```

### 2. Slow Weather Data Download

**Problem**: First run weather data download is slow

**Solution**:
- First run downloads data from Open-Meteo API (~10-30 seconds)
- Data automatically cached to `openmeteo_cache/` directory
- Subsequent runs load from cache, takes seconds

### 3. CUDA Out of Memory

**Problem**: `CUDA out of memory`

**Solution**: Reduce batch size
```python
CONFIG = {
    'computation': {
        'batch_size': 50,  # Default: 100
    }
}
```

### 4. Datetime Parsing Failure

**Problem**: `ValueError: time data does not match format`

**Solution**: Check if original CSV datetime format is `YYYYMMDDHHmmss`
```python
# Correct format example
20190228235236  # 2019-02-28 23:52:36
```

### 5. RealSceneDL Module Not Found

**Solution**: Install RealSceneDL or add it to Python path

```bash
# Option 1: Install in editable mode
cd /path/to/RealSceneDL
pip install -e .

# Option 2: Add to Python path
export PYTHONPATH="/path/to/RealSceneDL/src:$PYTHONPATH"
```

### 6. Invalid Footprint Data

**Solution**: Ensure your data has required fields

```python
import geopandas as gpd
gdf = gpd.read_file('buildings.geojson')
print(gdf.columns)  # Must include 'geometry' and 'height'
print(gdf.crs)      # Should be EPSG:4326 (WGS84)
```

### 7. Trajectory Data Format Error

**Ensure CSV contains** (no header before preprocessing):
- `datetime`: Timestamp (format: YYYYMMDDHHmmss)
- `vehicle_id`: Vehicle plate number
- `lng`: Longitude
- `lat`: Latitude
- `speed`: Speed
- `angle`: Vehicle heading angle (degrees, 0=North)
- `operation_status`: Operation status

## 🔬 Technical Details

### Shadow Calculation

- **Ray Tracing**: triro/OptiX GPU ray tracing for batch shadow detection
- **GPU Optimization**: Batch ray generation using PyTorch tensors
- **Sun Position**: pvlib-based solar position calculations

### PV Modeling

- **Irradiance Model**: Isotropic sky model
- **POA Calculation**: Manual vectorized POA formula implementation (direct + diffuse + reflected)
- **Temperature Model**: SAPM cell temperature model
- **Power Calculation**: Efficiency model with temperature correction

### POA (Plane of Array) Calculation Formula

```python
# Direct component: POA_direct = DNI × cos(AOI)
poa_direct = DNI × (cos(zenith) × cos(tilt) +
                    sin(zenith) × sin(tilt) × cos(azimuth_sun - azimuth_surf))

# Diffuse component (isotropic): POA_diffuse = DHI × (1 + cos(tilt)) / 2
poa_diffuse = DHI × (1 + cos(tilt)) / 2

# Reflected component: POA_reflected = GHI × albedo × (1 - cos(tilt)) / 2
poa_reflected = GHI × albedo × (1 - cos(tilt)) / 2

# Total POA
POA_global = poa_direct + poa_diffuse + poa_reflected
```

### Coordinate Systems

- **Input**: WGS84 (GPS coordinates)
- **Internal**: Local Cartesian coordinates
- **Transformation**: RealSceneDL coordinate utilities

### Batch Processing Strategy

- **Single-GPU Serial**: All trajectories processed serially on a single GPU
- **Internal GPU Parallelism**: Each trajectory uses GPU-parallel computation
- **Resource Sharing**: Building mesh and calculator shared across all trajectories

## 📚 Related Documentation

- [code/BATCH_PROCESSING_README.md](code/BATCH_PROCESSING_README.md) - Detailed batch processing guide
- [code/GPU环境配置指南.md](code/GPU环境配置指南.md) - GPU environment setup guide
- **pvlib-python**: https://pvlib-python.readthedocs.io/
- **Open-Meteo API**: https://open-meteo.com/
- **PyVista Ray Tracing**: https://docs.pyvista.org/
- **RealSceneDL**: Building footprint to 3D mesh conversion library

## 📄 License

This project is for research purposes only.

## 📧 Contact

For questions and issues, please contact the project maintainer.
