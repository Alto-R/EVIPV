# Vehicle Rooftop PV Generation Calculation Toolkit

A vehicle photovoltaic generation calculation system based on GPS trajectories and 3D building models, supporting batch processing for both taxis and buses.

English | **[中文](README.md)**

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Quick Start](#quick-start)
- [Complete Workflow](#complete-workflow)
  - [Taxi Processing Workflow](#taxi-processing-workflow)
  - [Bus Processing Workflow](#bus-processing-workflow)
- [Script Details](#script-details)
- [Configuration Guide](#configuration-guide)
- [Output Format](#output-format)
- [Performance Optimization](#performance-optimization)
- [FAQ](#faq)

---

## System Overview

### Key Features

- ✅ **Dual Vehicle Type Support**: Independent processing pipelines for taxis and buses
- ✅ **GPU Acceleration**: 8-10x performance boost using PyTorch + triro/OptiX
- ✅ **High-Precision Shadows**: GPU ray tracing for building shadow analysis
- ✅ **Intelligent Data Processing**: Automatic weather data fetching, coordinate transformation, parking interpolation
- ✅ **Batch Parallelism**: Multi-process preprocessing + GPU batch computation
- ✅ **Bus Route Analysis**: Automatic route statistics and representative trajectory selection

### System Requirements

**Required Dependencies:**
```bash
# Python 3.8+
pip install pandas numpy geopandas shapely
pip install pyvista trimesh pvlib-python pyproj pyyaml tqdm
```

**GPU Acceleration (Highly Recommended):**
```bash
# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For detailed GPU environment setup, see: [GPU环境配置指南.md](GPU环境配置指南.md)

---

## Quick Start

### 1. Prepare Data

```bash
# Directory structure
data/
├── shenzhen_building.geojson      # Building footprint data
├── taxi/
│   └── raw/                        # Raw taxi GPS data
└── bus/
    └── raw/                        # Raw bus GPS data
```

### 2. Preprocess Trajectories

**Taxi:**
```bash
python preprocess_taxi_trajectories.py
```

**Bus:**
```bash
python preprocess_bus_trajectories.py
```

### 3. Batch Calculate PV Generation

**Taxi:**
```bash
python batch_process_trajectories_taxi.py
```

**Bus:**
```bash
python batch_process_trajectories_bus.py
```

---

## Complete Workflow

### Taxi Processing Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data Preprocessing (preprocess_taxi_trajectories.py)         │
├─────────────────────────────────────────────────────────────────┤
│ Input: raw/*.csv (Raw GPS data, headerless)                     │
│ Output: processed/*_processed.csv                               │
│                                                                  │
│ Functions:                                                       │
│ - Add standard column names (datetime, vehicle_id, lng, lat,..) │
│ - Datetime parsing + timezone conversion (Asia/Shanghai)        │
│ - Data cleaning: deduplication, coordinate validation, angle    │
│ - Vehicle ID extraction and standardization                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Building Mesh Preparation (prepare_building_mesh_from_...)   │
├─────────────────────────────────────────────────────────────────┤
│ Input: shenzhen_building.geojson                                │
│ Output: building_mesh.ply                                       │
│                                                                  │
│ Functions:                                                       │
│ - Building footprint → 3D mesh conversion                       │
│ - Optional mesh subdivision (improve shadow accuracy)           │
│ - Parallel processing (multi-core CPU acceleration)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Batch PV Generation Calculation (batch_process_traj_taxi.py) │
├─────────────────────────────────────────────────────────────────┤
│ Input: processed/*_processed.csv + building_mesh.ply            │
│ Output: output_taxi/*_pv_generation.csv                         │
│                                                                  │
│ Sub-processes:                                                   │
│ ┌──────────────────────────────────────────────────────┐       │
│ │ 3.1 Data Preparation Stage (Multi-process parallel)   │       │
│ │   - Read trajectories                                  │       │
│ │   - Auto-fetch weather data (smart caching)            │       │
│ │   - Parking interpolation (resolution: 1 min)          │       │
│ │   - Month cloning (expand to all 12 months)            │       │
│ └──────────────────────────────────────────────────────┘       │
│                         ↓                                        │
│ ┌──────────────────────────────────────────────────────┐       │
│ │ 3.2 PV Calculation Stage (GPU batch computation)      │       │
│ │   - GPU ray tracing (building shadow detection)        │       │
│ │   - POA irradiance calculation (direct+diffuse+reflect)│       │
│ │   - Cell temperature model                             │       │
│ │   - DC/AC power calculation                            │       │
│ │   - Energy integration                                 │       │
│ └──────────────────────────────────────────────────────┘       │
│                                                                  │
│ Configuration:                                                   │
│ - panel_area: 2.2 m² (taxi rooftop area)                        │
│ - vehicle_height: 1.5 m                                         │
│ - max_vehicles: 1050 (processing limit)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Bus Processing Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data Preprocessing (preprocess_bus_trajectories.py)          │
├─────────────────────────────────────────────────────────────────┤
│ Input: raw/*.csv (Raw GPS data with busline_name column)        │
│ Output:                                                          │
│   - processed/*_processed.csv (with busline_name retained)      │
│   - busline_summary.csv (route statistics summary)              │
│   - representative_trajectories/*.csv (per-route representative)│
│                                                                  │
│ Functions:                                                       │
│ - Standardize format (8 columns, including busline_name)        │
│ - Datetime parsing + timezone conversion                        │
│ - Data cleaning + NaN filtering                                 │
│ - Bus route statistics:                                          │
│   * Identify all unique routes (e.g., 1549 routes)              │
│   * Count vehicles and records per route                        │
│ - Representative trajectory selection:                           │
│   * Select vehicle with most GPS records per route              │
│   * Output to separate folder                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Batch PV Generation Calculation (batch_process_traj_bus.py)  │
├─────────────────────────────────────────────────────────────────┤
│ Input: representative_trajectories/*.csv                        │
│ Output: output_bus/*_pv_generation.csv                          │
│                                                                  │
│ Configuration (differs from taxi):                               │
│ - panel_area: 25 m² (larger bus rooftop)                        │
│ - vehicle_height: 3.0 m (taller buses)                          │
│ - max_vehicles: None (process all 1549 routes)                  │
│                                                                  │
│ Other processes same as taxi                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Script Details

### Data Preprocessing Scripts

#### `preprocess_taxi_trajectories.py`

**Purpose**: Taxi GPS data preprocessing

**Functions**:
- Add standard column names (datetime, vehicle_id, lng, lat, speed, angle, operation_status)
- Parse datetime and convert to timezone-aware format (Asia/Shanghai)
- Data cleaning: deduplication, coordinate validation, angle filling
- Vehicle ID extraction and standardization

**Output Format**:
```csv
datetime,vehicle_id,lng,lat,speed,angle,operation_status
2019-08-15 10:45:30+08:00,Z8,114.002296,22.537403,0.0,169.56,0
```

**Configuration**:
```python
CONFIG = {
    'input_dir': '<your_raw_data_path>',
    'output_dir': '<your_processed_data_path>',
    'processing': {
        'chunk_size': 1000,        # Chunk size for large files
        'n_jobs': 10,              # Number of parallel processes
    }
}
```

**Run**:
```bash
python preprocess_taxi_trajectories.py
```

---

#### `preprocess_bus_trajectories.py`

**Purpose**: Bus GPS data preprocessing + route statistics + representative trajectory selection

**Functions**:
- Standardize format (8 columns, including busline_name)
- Parse datetime and convert to timezone-aware format (Asia/Shanghai)
- Data cleaning + NaN filtering
- Bus route statistics and representative trajectory selection

**Output**:

1. **Preprocessed Files** (`processed/*_processed.csv`):
```csv
datetime,vehicle_id,lng,lat,speed,angle,operation_status,busline_name
2019-08-15 10:45:30+08:00,B12345,114.002296,22.537403,0.0,169.56,0,M123
```

2. **Route Summary** (`busline_summary.csv`):
```csv
busline_name,vehicle_count,total_records,avg_records_per_vehicle
M123,25,125000,5000
M456,18,90000,5000
...
```

3. **Representative Trajectories** (`representative_trajectories/M123_representative_processed.csv`):
- Select vehicle with most GPS records per route

**Configuration**:
```python
CONFIG = {
    'input_dir': '<your_bus_raw_data_path>',
    'output_dir': '<your_bus_processed_data_path>',
    'processing': {
        'chunk_size': 100000,
        'n_jobs': 10,
    }
}
```

**Run**:
```bash
python preprocess_bus_trajectories.py
```

---

### Building Data Processing Scripts

#### `prepare_building_mesh_from_footprint.py`

**Purpose**: Convert building footprint data to 3D mesh

**Input Requirements**:
- Format: GeoJSON / Shapefile / GeoPackage
- Required fields:
  - `geometry`: Polygon (WGS84, EPSG:4326)
  - `height`: Building height (meters)

**Output**: `.ply` format trimesh file

**Configuration**:
```python
CONFIG = {
    'FOOTPRINT_PATH': '../data/shenzhen_building.geojson',
    'OUTPUT_MESH_PATH': '../data/shenzhen_building_mesh.ply',
    'GRID_SIZE': None,    # Mesh subdivision precision (meters), None=no subdivision
    'N_JOBS': 20,         # Number of parallel processes
    'BATCH_SIZE': None,   # Auto-allocate
}
```

**Run**:
```bash
python prepare_building_mesh_from_footprint.py
```

**Performance**:
- Single process: ~several minutes (depends on building count)

---

#### `fetch_irradiance_data.py`

**Purpose**: Manually fetch solar irradiance data (typically called automatically by batch scripts)

**Data Source**: Open-Meteo API

**Configuration**:
```python
CONFIG = {
    'LAT': 22.543099,
    'LON': 114.057868,
    'START_DATE': '2019-01-01',
    'END_DATE': '2020-01-01',
    'GRANULARITY': '1min',        # '1min' or '1hour'
    'SAVE_CSV': True,
    'OUTPUT_DIR': 'irradiance_data',
}
```

**Output**:
- Auto-cached: `openmeteo_cache/*.parquet`
- Optional CSV: `irradiance_data/*.csv`

**Run**:
```bash
python fetch_irradiance_data.py
```

---

### Batch Calculation Scripts

#### `batch_process_trajectories_taxi.py`

**Purpose**: Batch taxi PV generation calculation

**Input**:
- Trajectories: `processed/*_processed.csv`
- Buildings: `building_mesh.ply`

**Output**:
- Detailed results: `output_taxi/{vehicle_id}_pv_generation.csv`
- Summary report: `output_taxi/batch_summary.txt`

**Configuration**:
```python
CONFIG = {
    'location': {
        'name': 'Shenzhen',
        'lat': 22.543099,
        'lon': 114.057868,
    },
    'data_sources': {
        'building_mesh_path': '<your_building_mesh_path>',
        'trajectory_dir': '<your_processed_trajectory_path>',
    },
    'pv_system': {
        'panel_area': 2.2,              # Taxi rooftop area (m²)
        'panel_efficiency': 0.20,       # PV panel efficiency
        'tilt': 0,                      # Tilt angle (degrees)
        'vehicle_height': 1.5,          # Rooftop height (m)
    },
    'computation': {
        'clone_to_all_months': True,    # Clone to all 12 months
        'max_vehicles': 1050,           # Max vehicles to process
        'parking_threshold_minutes': 2, # Parking threshold
        'interpolation_resolution_minutes': 1,  # Interpolation resolution
        'num_prepare_workers': 5,       # Data preparation workers
        'gpu_batch_size': 100,          # GPU batch size
    },
    'output': {
        'output_dir': 'output_taxi',
    },
}
```

**Run**:
```bash
python batch_process_trajectories_taxi.py
```

**Processing Flow**:
1. **Parallel Data Preparation** (5 worker processes):
   - Read trajectories
   - Fetch weather data
   - Parking interpolation
   - Month cloning
   - Queue data

2. **GPU Batch Computation** (main process):
   - Retrieve data from queue
   - GPU ray tracing
   - PV generation calculation
   - Save results

**Performance** (NVIDIA RTX 3080):
- Single vehicle full-year data: ~5 seconds
- 1050 vehicles: ~1.5 hours (continuous)

---

#### `batch_process_trajectories_bus.py`

**Purpose**: Batch bus PV generation calculation

**Input**:
- Trajectories: `representative_trajectories/*_representative_processed.csv`
- Buildings: `building_mesh.ply`

**Output**:
- Detailed results: `output_bus/{busline}_pv_generation.csv`
- Summary report: `output_bus/batch_summary.txt`

**Configuration** (differences from taxi):
```python
CONFIG = {
    # ... (location and building_mesh_path same as taxi)

    'data_sources': {
        'trajectory_dir': '<your_bus_representative_trajectory_path>',
    },
    'pv_system': {
        'panel_area': 25,               # ✅ Larger bus rooftop
        'panel_efficiency': 0.20,
        'tilt': 0,
        'vehicle_height': 3.0,          # ✅ Taller buses
    },
    'computation': {
        'clone_to_all_months': True,
        'max_vehicles': None,           # ✅ No limit (process all 1549 routes)
        # ... (other parameters same as taxi)
    },
    'output': {
        'output_dir': 'output_bus',     # ✅ Separate output directory
    },
}
```

**Run**:
```bash
python batch_process_trajectories_bus.py
```

**Performance** (NVIDIA RTX 3080):
- Single route full-year data: ~0.5 seconds
- 1549 routes: ~10 minutes (continuous)

---

### Core Computation Modules

#### `pv_generation_pvlib.py`

**Purpose**: Base PV generation calculator (CPU version)

**Functions**:
- Solar position calculation (pvlib)
- GPS coordinate transformation (WGS84 → local Cartesian)
- Ray tracing (trimesh + triro/OptiX)
- POA irradiance calculation
- DC/AC power calculation

**Main Class**:
```python
class SolarPVCalculator:
    def __init__(self, lon_center, lat_center, building_mesh,
                 panel_area=2.0, panel_efficiency=0.22, ...):
        pass

    def calculate_generation(self, trajectory_df, irradiance_df):
        """Calculate PV generation"""
        pass
```

---

#### `pv_calculator_gpu.py`

**Purpose**: GPU-accelerated PV calculator

**Inherits From**: `SolarPVCalculator`

**GPU Optimizations**:
- PyTorch batch ray generation
- GPU parallel power calculation
- Pre-built coordinate transformers
- Batch processing optimization

**Main Class**:
```python
class GPUAcceleratedSolarPVCalculator(SolarPVCalculator):
    def __init__(self, *args, use_gpu=True, **kwargs):
        # Auto-detect CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Performance Boost**: CPU mode → GPU mode approximately **8-10x**

---

## Configuration Guide

### Taxi vs Bus Configuration Comparison

| Parameter | Taxi | Bus | Description |
|-----------|------|-----|-------------|
| `panel_area` | 2.2 m² | 25 m² | Larger bus rooftop area |
| `vehicle_height` | 1.5 m | 3.0 m | Taller buses |
| `max_vehicles` | 1050 | None | Bus processes all routes |
| `trajectory_dir` | `taxi/processed/` | `bus/representative_trajectories/` | Different data sources |
| `output_dir` | `output_taxi/` | `output_bus/` | Separate outputs |

### GPU Memory Optimization

| VRAM | Recommended batch_size | Description |
|------|------------------------|-------------|
| 4GB | 20-50 | Small batches |
| 6GB | 50-100 | Medium batches |
| 8GB | 100-200 | Large batches |
| 12GB+ | 200-500 | Extra-large batches |

**If encountering CUDA out of memory**:
```python
CONFIG = {
    'computation': {
        'gpu_batch_size': 50,  # Reduce batch size
    }
}
```

### Parking Interpolation Configuration

```python
CONFIG = {
    'computation': {
        'parking_threshold_minutes': 2,          # Parking threshold
        'interpolation_resolution_minutes': 1,   # Interpolation resolution
    }
}
```

**Explanation**:
- Parking threshold: Consecutive points with interval > threshold and same position → parking detected
- Interpolation resolution: Time interval for interpolation during parking periods
- PV generation calculated continuously during parking (sun position changes)

---

## Output Format

### Detailed Results CSV

**File**: `output_taxi/{vehicle_id}_pv_generation.csv`

| Column | Description | Unit | Example |
|--------|-------------|------|---------|
| `datetime` | Timestamp | - | `2019-08-15 10:50:11+08:00` |
| `lng`, `lat` | GPS coordinates | degrees | `114.0023`, `22.5374` |
| `speed` | Speed | km/h | `0.0` (parked) |
| `angle` | Vehicle heading | degrees | `169.56` (SSE) |
| `month` | Month | 1-12 | `8` |
| `is_shaded` | Shadow status | 0/1/2/3/4 | `0`=unshaded, `1-4`=varying shade levels |
| `poa_global` | Total plane-of-array irradiance | W/m² | `590.83` |
| `poa_direct` | Direct irradiance | W/m² | `364.55` |
| `poa_diffuse` | Diffuse irradiance | W/m² | `243.77` |
| `poa_reflected` | Reflected irradiance | W/m² | `234.02` |
| `cell_temp` | Cell temperature | °C | `40.57` |
| `dc_power` | DC power | W | `236.87` |
| `ac_power` | AC power | W | `226.06` (considering inverter efficiency) |
| `delta_t_seconds` | Time interval | seconds | `60.0` |
| `energy_kwh` | Energy | kWh | `0.00395` |

### Batch Processing Summary

**File**: `output_taxi/batch_summary.txt`

```
================================================================================
  Batch Processing Summary Report
================================================================================

Processing completed: 2025-01-15 10:30:45
Total computation time: 2 days 15 hours 30 minutes

--------------------------------------------------------------------------------
  Overall Statistics
--------------------------------------------------------------------------------
Successfully processed vehicles: 1050
Total energy generation: 15,234.56 kWh
Average per vehicle: 14.51 kWh

--------------------------------------------------------------------------------
  Detailed Statistics (by vehicle)
--------------------------------------------------------------------------------
Vehicle   Trajectory   Total Energy   Avg Power   Shadow    Computation
ID        Points       (kWh)          (W)         Rate(%)   Time
------------------------------------------------------------------------
Z8        71,300       18.45          245.2       35.2      8m15s
0G        125,531      25.67          218.9       42.1      12m38s
...
```

---

## Performance Optimization

### Multi-Process Parallel Optimization

**Data Preparation Stage** (CPU-intensive):
```python
CONFIG = {
    'computation': {
        'num_prepare_workers': 5,  # Adjust based on CPU cores
    }
}
```

**Recommendations**:
- CPU cores ≤ 8: Use 3-5 workers
- CPU cores > 8: Use 5-10 workers

### GPU Batch Processing Optimization

**GPU Computation Stage**:
```python
CONFIG = {
    'computation': {
        'gpu_batch_size': 100,  # Adjust based on VRAM
    }
}
```

**Tuning Strategy**:
1. Monitor GPU memory during runtime: `nvidia-smi -l 1`
2. If VRAM utilization < 80%: Increase batch_size
3. If encountering OOM: Decrease batch_size

### Time Resolution Trade-offs

```python
CONFIG = {
    'computation': {
        'interpolation_resolution_minutes': 1,  # 1/5/10 minutes
    }
}
```

| Resolution | Accuracy | Computation | Use Case |
|------------|----------|-------------|----------|
| 1 minute | Highest | Largest | Research-grade precision |
| 5 minutes | High | Medium | General analysis |
| 10 minutes | Medium | Small | Quick estimation |

---

## Related Documentation

- [GPU环境配置指南.md](GPU环境配置指南.md) - Detailed GPU environment setup
- **pvlib-python**: https://pvlib-python.readthedocs.io/
- **Open-Meteo API**: https://open-meteo.com/
- **PyVista**: https://docs.pyvista.org/
- **trimesh**: https://trimsh.org/

---

## License

This project is for research purposes only.

---

## Contact

For questions or suggestions, please contact the project maintainer.
