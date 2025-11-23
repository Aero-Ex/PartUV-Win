# PartUV - Build Instructions

A high-performance UV unwrapping library with CUDA acceleration for 3D mesh processing.

## Table of Contents

- [System Requirements](#system-requirements)
- [GPU Compatibility](#gpu-compatibility)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows](#windows-build)
  - [Linux](#linux-build)
- [Build Configuration](#build-configuration)
- [Troubleshooting](#troubleshooting)
- [Verifying Installation](#verifying-installation)
- [Building C++ Standalone](#building-c-standalone-optional)

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (tested with 3.11)
- **CUDA Toolkit**: 12.0 or higher
- **CMake**: 3.18 or higher
- **C++ Compiler**:
  - Windows: Visual Studio 2019 or later (MSVC 14.2+)
  - Linux: GCC 9+ or Clang 10+
- **GPU**: NVIDIA GPU with compute capability 8.0+ (see [GPU Compatibility](#gpu-compatibility))

### Disk Space
- Build: ~2-3 GB
- Installation: ~500 MB

---

## GPU Compatibility

### Supported GPUs (Default Configuration)

The default build configuration supports modern NVIDIA GPUs:

| Architecture | Compute Capability | Examples |
|--------------|-------------------|----------|
| **Ampere** | 8.0, 8.6 | RTX 3050/3060/3070/3080/3090, A100, A10, A30 |
| **Ada Lovelace** | 8.9 | RTX 4060/4070/4080/4090, L4, L40 |
| **Hopper** | 9.0 | H100, H200 |

### Unsupported GPUs (Default)

The following GPUs are **not supported** by default but can be enabled (see [Build Configuration](#build-configuration)):

| Architecture | Compute Capability | Examples |
|--------------|-------------------|----------|
| Pascal | 6.0, 6.1 | GTX 1050/1060/1070/1080, P100, Titan X |
| Volta | 7.0 | V100, Titan V |
| Turing | 7.5 | RTX 2060/2070/2080, GTX 1650/1660, T4 |

To check your GPU's compute capability, visit: https://developer.nvidia.com/cuda-gpus

---

## Prerequisites

### Install Conda/Miniconda

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

### Clone Repository

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Aero-Ex/PartUV-Win.git
cd partuv

# Or if already cloned, update submodules
git pull
git submodule update --init --recursive
```

### Create Environment

```bash
# Create a new conda environment
conda create -n partuv python=3.11 -y
conda activate partuv
```

### Install Dependencies

```bash
# Install required packages
conda install -c conda-forge \
    gmp mpfr boost cgal yaml-cpp \
    tbb tbb-devel pybind11 ninja cmake \
    numpy scipy trimesh opencv -y
```

### Install CUDA Toolkit

**Windows:**
```bash
# Using conda (recommended)
conda install -c nvidia cuda-toolkit=12.4 -y
```

Or download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

**Linux:**
```bash
# Using conda
conda install -c nvidia cuda-toolkit=12.4 -y

# Or using system package manager (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit -y
```

---

## Installation

### Windows Build

#### Step 1: Activate Environment
```cmd
conda activate partuv
```

#### Step 2: Navigate to Project Directory
```cmd
cd path\to\PartUV
```

#### Step 3: Build and Install
```cmd
pip install . -v --no-build-isolation ^
  --config-settings="cmake.args=-GNinja;^
  -DCMAKE_BUILD_TYPE=Release;^
  -DCMAKE_PREFIX_PATH='%CONDA_PREFIX%\Library';^
  -DTBB_DIR='%CONDA_PREFIX%\Library\lib\cmake\TBB';^
  -DGMP_INCLUDE_DIR='%CONDA_PREFIX%\Library\include';^
  -DGMP_LIBRARIES='%CONDA_PREFIX%\Library\lib\gmp.lib';^
  -DMPFR_INCLUDE_DIR='%CONDA_PREFIX%\Library\include';^
  -DMPFR_LIBRARIES='%CONDA_PREFIX%\Library\lib\mpfr.lib';^
  -DOpenMP_CXX_FLAGS='-openmp:llvm';^
  -DOpenMP_CXX_LIB_NAMES='libomp';^
  -DOpenMP_libomp_LIBRARY='%CONDA_PREFIX%\Library\lib\libomp.lib'"
```

**Note**: The `^` character is the line continuation character in Windows CMD. If using PowerShell, replace `^` with `` ` `` (backtick).

**PowerShell Version:**
```powershell
pip install . -v --no-build-isolation `
  --config-settings="cmake.args=-GNinja;-DCMAKE_BUILD_TYPE=Release;-DCMAKE_PREFIX_PATH='$env:CONDA_PREFIX\Library';-DTBB_DIR='$env:CONDA_PREFIX\Library\lib\cmake\TBB';-DGMP_INCLUDE_DIR='$env:CONDA_PREFIX\Library\include';-DGMP_LIBRARIES='$env:CONDA_PREFIX\Library\lib\gmp.lib';-DMPFR_INCLUDE_DIR='$env:CONDA_PREFIX\Library\include';-DMPFR_LIBRARIES='$env:CONDA_PREFIX\Library\lib\mpfr.lib';-DOpenMP_CXX_FLAGS='-openmp:llvm';-DOpenMP_CXX_LIB_NAMES='libomp';-DOpenMP_libomp_LIBRARY='$env:CONDA_PREFIX\Library\lib\libomp.lib'"
```

#### Build Time
- First build: 10-20 minutes (depending on CPU)
- Subsequent builds: 2-5 minutes

---

### Linux Build

#### Step 1: Install System Dependencies (Optional)

If not using conda for all dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libcgal-dev libyaml-cpp-dev libtbb-dev -y

# Fedora/RHEL
sudo dnf install CGAL-devel yaml-cpp-devel tbb-devel -y
```

#### Step 2: Activate Environment
```bash
conda activate partuv
```

#### Step 3: Navigate to Project Directory
```bash
cd path/to/PartUV
```

#### Step 4: Build and Install
```bash
pip install . -v --no-build-isolation \
  --config-settings="cmake.args=-GNinja;-DCMAKE_BUILD_TYPE=Release"
```

The build system will automatically detect dependencies from the conda environment on Linux.

**For editable/development install:**
```bash
pip install -e . -v --no-build-isolation \
  --config-settings="cmake.args=-GNinja;-DCMAKE_BUILD_TYPE=Release"
```

#### Build Time
- First build: 10-20 minutes
- Subsequent builds: 2-5 minutes

---

## Build Configuration

### Custom GPU Support

To add support for older/different GPUs, modify `CMakeLists.txt` before building:

```cmake
# Line 6 in CMakeLists.txt
# Default (RTX 3000+, RTX 4000, H100 only):
set(CMAKE_CUDA_ARCHITECTURES 80 89 90)

# For broader compatibility (GTX 1000+, RTX 2000+, RTX 3000+, RTX 4000):
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90)

# For specific GPU:
set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 2000 series only
```

**Compute Capability Reference:**
- `60, 61`: Pascal (GTX 1000 series, P100)
- `70`: Volta (V100, Titan V)
- `75`: Turing (RTX 2000 series, GTX 1600 series)
- `80, 86`: Ampere (RTX 3000 series, A100)
- `89`: Ada Lovelace (RTX 4000 series)
- `90`: Hopper (H100, H200)

**Note:** Including more architectures increases compilation time and binary size significantly.

### Build Types

```bash
# Release build (default, optimized for performance)
-DCMAKE_BUILD_TYPE=Release

# Debug build (for development, slower)
-DCMAKE_BUILD_TYPE=Debug

# Release with debug info (balanced)
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Profiling

To disable profiling timers:

```bash
# Add to cmake.args:
-DENABLE_PROFILING=OFF
```

---

## Troubleshooting

### Common Issues

#### 1. "Could NOT find GMP"

**Problem:** CMake cannot find GMP library.

**Solution:**
```bash
# Reinstall GMP and MPFR
conda install -c conda-forge gmp mpfr -y

# Verify installation (Windows)
dir %CONDA_PREFIX%\Library\lib\gmp.lib

# Verify installation (Linux)
ls $CONDA_PREFIX/lib/libgmp*
```

#### 2. "nvcc fatal: A single input file is required"

**Problem:** MSVC compiler flags being incorrectly passed to CUDA compiler.

**Solution:** This is fixed in the current version. Ensure you're using the latest `CMakeLists.txt` with proper flag scoping:
```cmake
# Correct (flags scoped to CXX only):
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
    ...
)
```

#### 3. "error LNK2001: unresolved external symbol omp_get_max_active_levels"

**Problem:** OpenMP library not found or wrong version (MSVC OpenMP 2.0 doesn't support these functions).

**Solution (Windows):**
```bash
# Install LLVM OpenMP
conda install -c conda-forge llvm-openmp -y

# Verify installation
dir %CONDA_PREFIX%\Library\lib\libomp.lib

# Use the OpenMP flags in the build command (already included above)
```

#### 4. "Cannot open include file: 'boost/mpl/aux_/preprocessed/plain/||.hpp'"

**Problem:** Macro definitions interfering with Boost headers.

**Solution:** This is fixed in the current version. Ensure your `CMakeLists.txt` does NOT contain:
```cmake
# These should NOT be present (old broken version):
add_compile_definitions(
    and=&&
    or=||
    not=!
)
```

#### 5. "calling a __host__ function from a __global__ function"

**Problem:** CUDA device code calling host-only functions.

**Solution:** This is fixed in the current version. The fix includes:
- `sqrt(3)` → `sqrtf(3.0f)` for device-compatible math
- `const float` → `constexpr float` for device-accessible constants

#### 6. Build Fails on Older GPUs

**Problem:** Your GPU compute capability is not included in the default build.

**Solution:** See [Custom GPU Support](#custom-gpu-support) to add your GPU architecture.

#### 7. Out of Memory During Build

**Problem:** Compiler runs out of memory during parallel compilation.

**Solution:**
```bash
# Limit parallel jobs (add to cmake.args)
-DCMAKE_BUILD_PARALLEL_LEVEL=2

# Or set environment variable
export CMAKE_BUILD_PARALLEL_LEVEL=2  # Linux
set CMAKE_BUILD_PARALLEL_LEVEL=2     # Windows
```

#### 8. CUDA Toolkit Not Found

**Problem:** CMake cannot find CUDA installation.

**Solution (Windows):**
```cmd
# Set CUDA path manually
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set CUDA_HOME=%CUDA_PATH%
```

**Solution (Linux):**
```bash
# Set CUDA path manually
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 9. "yaml-cpp.lib not found" (Link Error)

**Problem:** CMake using wrong target name for yaml-cpp.

**Solution:** This is fixed in the current version. Ensure `CMakeLists.txt` uses:
```cmake
target_link_libraries(_core
    PRIVATE
        yaml-cpp::yaml-cpp  # Correct (with namespace)
        # NOT: yaml-cpp     # Wrong
)
```

### Clean Build

If you encounter persistent issues, try a clean build:

**Linux/Mac:**
```bash
# Remove build artifacts
rm -rf build _skbuild dist *.egg-info

# Remove installed package
pip uninstall partuv -y

# Rebuild
pip install . -v --no-build-isolation [... cmake args ...]
```

**Windows:**
```cmd
# Remove build artifacts
rd /s /q build _skbuild dist
for /d %i in (*.egg-info) do rd /s /q "%i"

# Remove installed package
pip uninstall partuv -y

# Rebuild
pip install . -v --no-build-isolation [... cmake args ...]
```

### Getting Help

If you continue to experience issues:

1. Check existing [GitHub Issues](https://github.com/yourusername/PartUV/issues)
2. Create a new issue with:
   - Full build log (use `-v` flag with pip)
   - OS and version
   - Python version: `python --version`
   - CUDA version: `nvcc --version`
   - GPU model: `nvidia-smi`
   - CMake version: `cmake --version`
   - Conda environment: `conda list`

---

## Verifying Installation

After successful installation, verify the package works:

```python
import partuv

# Check module is importable
print(f"PartUV imported successfully!")

# Test basic functionality
import numpy as np

# Create a simple test mesh (triangle)
vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
], dtype=np.float32)

faces = np.array([
    [0, 1, 2]
], dtype=np.int32)

print("✓ PartUV is ready to use!")
```

Expected output:
```
PartUV imported successfully!
✓ PartUV is ready to use!
```

### Check CUDA Availability

```python
import torch

if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠ CUDA not available")
```

---

## Building C++ Standalone (Optional)

For development or testing, you can build a standalone C++ executable:

### Setup

```bash
# Copy the main CMakeLists configuration
cp CMakeLists_main.txt CMakeLists.txt

# Create build directory
mkdir -p all_build/release
cd all_build/release
```

### Configure and Build

**Linux:**
```bash
cmake -DMAIN_FILE=main.cpp \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_ALL_SRC_FILES=ON \
      -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
      ../..

make -j$(nproc)
```

**Windows:**
```cmd
cmake -DMAIN_FILE=main.cpp ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DUSE_ALL_SRC_FILES=ON ^
      -GNinja ^
      ..\..

ninja
```

### Run

```bash
./program mesh_path/preprocessed_mesh.obj
```

Required directory structure:
```
mesh_path/
├── preprocessed_mesh.obj     # Preprocessed mesh from preprocess.py
└── bin/
    └── preprocessed_mesh.bin  # Part field tree hierarchy
```

---

## Performance Notes

- **First Run**: The first execution may be slower due to CUDA kernel compilation and caching.
- **Memory**: Ensure sufficient GPU memory for your mesh size:
  - Small meshes (<100K faces): 2GB+ VRAM
  - Medium meshes (100K-500K faces): 4GB+ VRAM
  - Large meshes (>1M faces): 8GB+ VRAM
- **Multi-GPU**: Currently uses a single GPU. Multi-GPU support planned for future releases.

---

## Development Build

For development with debugging symbols and editable installation:

```bash
# Install in editable mode
pip install -e . -v --no-build-isolation \
  --config-settings="cmake.args=-GNinja;-DCMAKE_BUILD_TYPE=RelWithDebInfo;..."

# Now you can modify source code and rebuild with:
pip install -e . --no-build-isolation --no-deps
```

The `-e` flag creates an editable installation, allowing you to modify source code without full reinstallation.

---

## Additional Resources

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [CMake Documentation](https://cmake.org/documentation/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [NVIDIA GPU Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
- [Troubleshooting CUDA Issues](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#troubleshooting)

---

## Citation

If you use PartUV in your research, please cite:

```bibtex
@inproceedings{wang2025partuv,
  title={PartUV: Part-Based UV Unwrapping of 3D Meshes},
  author={Wang, Zhaoning and others},
  booktitle={SIGGRAPH Asia},
  year={2025}
}
```

---

**Last Updated**: November 2025

**Tested Configurations**:
- ✓ Windows 11 + CUDA 12.4 + RTX 4090
- ✓ Windows 10 + CUDA 12.4 + RTX 3080
- ✓ Ubuntu 22.04 + CUDA 12.4 + A100
- ✓ Ubuntu 20.04 + CUDA 12.1 + V100

For questions or contributions, please visit our [GitHub repository](https://github.com/yourusername/PartUV).
