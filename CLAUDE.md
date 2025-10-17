# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealSceneDL is a specialized Python library for processing 3D real-world scene data, focusing on Google Map 3D Tiles data acquisition, processing, and analysis. The library provides comprehensive tools for model processing, solar analysis, format conversion, and coordinate system transformations.

## Essential Commands

### Installation and Setup
```bash
# Install in editable mode (changes reflect immediately)
pip install -e .

# Install with documentation dependencies
pip install -e ".[docs]"
```

### Apple Silicon Mac Compatibility (Critical)
This project requires x86_64 architecture due to dependency limitations. On Apple Silicon Macs:
```bash
export CONDA_SUBDIR=osx-64
conda create -n realscenedl_intel python=3.8
conda activate realscenedl_intel
# Verify: python -c "import platform; print(platform.machine())"  # Should output: x86_64
```

### Testing
```bash
# Run all tests
pytest

# Skip slow/network/API tests for quick verification
pytest -m "not slow and not network and not api"

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
```

### Documentation
```bash
# Build and clean documentation
./scripts/deploy_docs.sh build

# Serve documentation locally
./scripts/deploy_docs.sh serve

# Manual MkDocs commands
mkdocs build --clean
mkdocs serve
```

## Architecture and Code Organization

### Core Structure
- **src/RealSceneDL/**: Main package source
  - **core/**: Data structures (tile.py, wgs84.py, bounding_volume.py)
  - **io/**: Input/output operations (download.py)
  - **processing/**: Data processing (merge.py, filter.py, flatten.py)
  - **analysis/**: Solar radiation and PV potential analysis
  - **rendering/**: Visualization and point cloud generation
  - **convert/**: Format conversion (B3DM, OSGB, GLB, PLY)
  - **coordinates/**: WGS84 ↔ 3D Tiles coordinate transformations
  - **footprint/**: Building footprint to 3D mesh conversion
  - **utils/**: General utilities (GLB fixing, scene merging)
  - **_internal/**: Internal modules (Draco compression support)

### Key Design Patterns
- **Dynamic Patching**: Automatically patches trimesh library on import to support Draco-compressed GLTF/GLB files
- **Modular Architecture**: Clear separation of concerns across functional domains
- **Coordinate System Abstraction**: Handles complex WGS84 ↔ Cartesian transformations transparently

### Dependencies and Tech Stack
- **Core**: trimesh, numpy, pandas, geopandas, shapely
- **3D Processing**: pygltflib, pydracogltf, py3dtiles, pyvista
- **Solar Analysis**: pvlib, suncalc
- **Documentation**: MkDocs with Material theme
- **Testing**: pytest with comprehensive marker system

## Development Guidelines

### Environment Setup
- Always use x86_64 environment on Apple Silicon Macs
- Configure Google Maps API key in `.env` file (never commit)
- Install in editable mode for immediate code reflection

### Task Completion Workflow
1. **Test**: Run `pytest -m "not slow and not network and not api"` for quick verification
2. **Full Testing**: Run `pytest` for comprehensive testing before major changes
3. **Documentation**: Rebuild with `./scripts/deploy_docs.sh build` if docstrings changed
4. **Import Verification**: Test `import RealSceneDL as rsdl` works correctly

### Code Conventions
- Python 3.8+ compatibility required
- Chinese comments and documentation are standard
- Use pytest markers: `unit`, `integration`, `slow`, `api`, `network`
- Follow modular organization patterns established in existing code
- Ensure no sensitive data (API keys) in source code

### Special Considerations
- **Trimesh Patch**: Changes to `_internal/gltf_with_draco.py` affect dynamic patching behavior
- **Coordinate Systems**: Validation required for coordinate transformation changes
- **3D Processing**: Test GLB/GLTF changes with actual 3D files
- **Apple Silicon**: Always verify x86_64 environment when developing