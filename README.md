
<h1 align="center">PartUV: Part-Based UV Unwrapping of 3D Meshes</h1> 
<h3 align="center">SIGGRAPH Asia 2025</h3> 

<p align="center">
<a href="https://arxiv.org/abs/2511.16659"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://www.zhaoningwang.com/PartUV"><img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project Page"></a>
</p>

Official implementation for ***PartUV: Part-Based UV Unwrapping of 3D Meshes***.
<p align="center"><img src="doc/partuv_teaser.png" width="100%"></p>


<!-- TOC -->
<details open>
  <summary><h1>📑 Table of Contents</h1></summary>


- [🚧 TODO List](#-todo-list)
- [🛠️ Installation](#-installation)
  - [PartUV (for UV Unwrapping)](#partuv-for-uv-unwrapping)
  - [Packing with bpy (optional)](#packing-with-bpy-optional)
- [🚀 Demo](#-demo)
  - [TL;DR](#tldr)
  - [Step 1: UV Unwrapping](#step-1-uv-unwrapping)
  - [Step 2: Packing](#step-2-packing)
  - [Part-Based Packing with UVPackMaster](#part-based-packing-with-uvpackmaster)
- [📊 Benchmarking](#-benchmarking-)
- [🧱 Building from Source](#-building-from-source)
- [🐛 Known Issues](#-known-issues)
- [🔧 Common Problems](#-common-problems)
- [🍀 Acknowledgement](#-acknowledgement)
- [🎓 BibTeX](#-bibtex)

</details>
<!-- /TOC -->





# 🚧 TODO List 
- [ ] Resolve the handling of non-2-manifold meshes, see [Known Issues](#-known-issues)
- [ ] Release benchmark code and data
- [ ] Multi-atlas packing with uvpackmaster
- [ ] Blender plugin for PartUV

# PartUV - Windows Build Instructions

**PartUV** is a high-performance UV unwrapping library with CUDA acceleration.

> ⚠️ **CRITICAL WINDOWS NOTICE**
> This project was originally designed for Linux. Building on Windows requires specific **Visual Studio Toolset versions**, **Conda dependencies**, and **minor code modifications**.
>
> Please follow this guide **exactly** to avoid the common `cudafe++` crashes and missing library errors.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Visual Studio Setup (Crucial)](#2-visual-studio-setup-crucial)
3. [Environment & Dependencies](#3-environment--dependencies)
4. [Fixing Source Code](#4-fixing-source-code)
5. [Modifying CMakeLists.txt](#5-modifying-cmakeliststxt)
6. [Building & Installing](#6-building--installing)
7. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Ensure you have the following installed:
*   **Git for Windows**
*   **Miniconda or Anaconda** (Python 3.11 recommended)
*   **CUDA Toolkit 12.4** (or 12.1)
*   **Visual Studio 2022 Build Tools** (Desktop C++ Workload)

---

## 2. Visual Studio Setup (Crucial)

**The Issue:** The latest Visual Studio 2022 compiler (v19.4x / Toolset 14.4x) creates C++ headers that are currently **incompatible** with CUDA 12.4, causing `nvcc` / `cudafe++` to crash with `ACCESS_VIOLATION`.

**The Fix:** You **must** install and use the older **v14.38** toolset.

1.  Open **Visual Studio Installer**.
2.  Click **Modify** next to your VS 2022 installation.
3.  Go to the **Individual Components** tab.
4.  Search for: `14.38`
5.  Check the box: **"MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.38-17.8)"**.
6.  Click **Modify** to install it.

---

## 3. Environment & Dependencies

We will use Conda to handle the difficult C++ libraries (CGAL, TBB) to avoid compiling them manually.

### A. Create Environment
Open your terminal and run:
```cmd
conda create -n partuv python=3.11
conda activate partuv
```

### B. Install Binary Dependencies
Install the build tools and libraries. **Note:** `tbb-devel` is required for the headers.
```cmd
conda install -c conda-forge cgal tbb tbb-devel pybind11 ninja cmake
pip install scikit-build-core
```

### C. Download Source & External Libraries
The project uses "Git Submodules" for dependencies like Eigen and LibIGL. If you downloaded a ZIP from GitHub, these folders are empty. We must fill them manually.

Run these commands in your project root:
```cmd
cd extern

# Clean potentially broken folders
rmdir /s /q eigen-3.4.0 libigl json stb OpenABF

# Download dependencies manually to ensure they exist
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git eigen-3.4.0
git clone --depth 1 https://github.com/libigl/libigl.git libigl
git clone --depth 1 https://github.com/nlohmann/json.git json
git clone --depth 1 https://github.com/nothings/stb.git stb
git clone --depth 1 https://github.com/EricWang12/OpenABF.git OpenABF

cd ..
```

---

## 4. Fixing Source Code

The original code uses Linux-specific logic that Visual Studio does not accept. You must edit **one file**.

**File:** `src/UnwrapMerge.cpp`  
**Line:** ~849 (Search for `#pragma omp parallel for`)

**Change this:**
```cpp
for (std::size_t k = 0; k < mergePairs.size(); ++k)
```
**To this:**
```cpp
// Cast size to int because MSVC OpenMP does not support unsigned variables
for (int k = 0; k < static_cast<int>(mergePairs.size()); ++k)
```

---

## 5. Modifying CMakeLists.txt

To fix the `error C2065: 'not': undeclared identifier` and missing math constants, replace the MSVC section in `CMakeLists.txt` with this block:

```cmake
if(MSVC)
    # Windows / Visual Studio Flags
    # /permissive- fixes 'not', 'and', 'or' keywords
    # /Zc:__cplusplus forces correct C++ version reporting
    add_compile_options(/O2 /fp:fast /permissive- /Zc:__cplusplus)
    
    # Fix compatibility definitions
    add_compile_definitions(
        _USE_MATH_DEFINES 
        NOMINMAX 
        M_PI=3.14159265358979323846
        _HAS_STD_BYTE=0        # Fixes ambiguous symbol 'byte' error
        and=&&                 # Manually define linux keywords if permissive- fails
        or=||
        not=!
    )

    # Suppress common NVCC warnings on Windows
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd4819\"") 
else()
```

---

## 6. Building & Installing

Perform the final build using the **x64 Native Tools Command Prompt for VS 2022** (Run as Administrator).

### Step 1: Activate the Compatible Toolset (v14.38)
Run this command to force the terminal to use the older compiler version we installed earlier.
```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.38
```
*Check:* Run `cl`. Output must start with **Version 19.38...**.

### Step 2: Clean Previous Attempts
```cmd
rmdir /s /q build
rmdir /s /q _skbuild
rmdir /s /q partuv.egg-info
```

### Step 3: Install
Run this exact command. It disables the internal profiler (which causes issues on Windows) and points CMake to your Conda libraries.

```cmd
pip install . -v --no-build-isolation --config-settings="cmake.args=-GNinja;-DCMAKE_PREFIX_PATH='%CONDA_PREFIX%\Library';-DTBB_DIR='%CONDA_PREFIX%\Library\lib\cmake\TBB';-DENABLE_PROFILING=OFF"
```

---

## Troubleshooting

| Error | Cause | Solution |
| :--- | :--- | :--- |
| **`fatal error C1083: Cannot open include file: 'Eigen/Core'`** | `extern` folders are empty. | Run the manual `git clone` commands in Section 3C. |
| **`nvcc error : 'cudafe++' died with status 0xC0000005`** | Using VS Toolset v14.4x (too new). | Downgrade terminal to v14.38 using the `vcvarsall.bat` command in Section 6. |
| **`CMake Error: Required library TBB not found`** | Missing development headers. | Run `conda install -c conda-forge tbb-devel`. |
| **`error C2065: 'not': undeclared identifier`** | MSVC doesn't support ISO keywords by default. | Apply the `CMakeLists.txt` patch in Section 5. |
| **`error C3016: index variable in OpenMP ... signed integral type`** | Code uses `size_t` in parallel loops. | Apply the source code fix in Section 4. |
| **`EasyProfiler ... libeasy_profiler.so not found`** | Linux binaries included in repo. | Use `-DENABLE_PROFILING=OFF` in the install command. |

Download the PartField checkpoint from [PartField](https://github.com/nv-tlabs/PartField):

```bash
wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt ./
```

## Packing with bpy (optional)

```bash
# For Python 3.11+
pip install bpy
# For Python 3.10
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
```

---




# 🚀 Demo

## TL;DR

```bash
python demo/partuv_demo.py --mesh_path {input_mesh_path} --save_visuals
```

<!-- # Step 1: UV unwrapping  
# Step 2: Pack the UV results with bpy
python -m pack.pack --partuv_output_path {partuv_output_folder} --save_visuals -->

---

## Step 1: UV Unwrapping

The demo takes a 3D mesh (e.g., .obj or .glb) as input and outputs the mesh with unwrapped UVs in a non-packed format.

### Input Requirement

We recommend using meshes without 3D self-intersections and non-2-manifold edges, as they may result in highly fragmented UVs.

### Preprocessing

The input mesh is first preprocessed, including:

* Mesh repair (e.g., merging nearby vertices, fixing non–2-manifold meshes, etc.)
* (Optional) Exporting the mesh to a `.obj` file
* Running PartField to obtain the hierarchical part tree

### Unwrapping
We then call our pre-built pip wheels for unwrapping. Two main API versions are provided:

* **`pipeline_numpy`**: The default version. It takes mesh NumPy arrays (`V` and `F`), the PartField dictionary (a hierarchical tree), a configuration file path, and a distortion threshold as input. Note that the distortion threshold specified here will override the value defined in the configuration file.
* **`pipeline`**: Similar to `pipeline_numpy`, but it takes file paths as input and performs I/O operations directly from disk.

### Output Results
Both APIs save the results to the output folder.
The final mesh with unwrapped UVs is saved as `final_components.obj`.
Each chart is flattened to a unit square, but inter-chart arrangement is not yet solved.

Individual parts are also saved as `part_{i}.obj`, which can be used with UVPackMaster to produce part-based UV packing (where charts belonging to the same part are grouped nearby). See the later section for more details.

The saving behavior can be configured in the [`save_results`](demo/partuv_demo.py#L119) function.

If you specify the `--pack_method` flags, the code will pack the UVs and save the final mesh in `final_packed.obj`.

### Hyperparameters

By default, the API reads all hyperparameters from `config/config.yaml`.
See [config.md](doc/config.md) for more details on hyperparameters and usage examples for customizing them to suit your needs.

---

## Step 2: Packing

The unwrapping API outputs UVs in a non-packed format.
You can pack all UV charts together to create a UV map for the input mesh. Two packing methods are supported:

* **`blender`**: The default packing method. We provide a script (`pack/pack_blender.py`) that uses `bpy` for packing, which is called by default in the demo file.
* **`uvpackmaster`**: A paid Blender add-on. We use this to achieve part-based packing (charts from the same part are packed close together) or automatic multi-atlas packing. Please see more details below.

---

## Part-Based Packing with UVPackMaster

In our results, we include both **part-based packing** (where charts from the same part are packed close together) and **automatic multi-atlas packing** (given *N* desired tiles, parts are assigned to tiles according to the hierarchical part tree).

These results are packed using [UVPackMaster](https://uvpackmaster.com/), which unfortunately is a paid tool. We provide scripts to pack the UVs with UVPackMaster.

### Installation

1. **Install BlenderProc:**
   We use BlenderProc to run this add-on within Blender. Please follow the instructions in the [BlenderProc repository](https://github.com/DLR-RM/blenderproc) to install it.

   ```bash
   pip install blenderproc
   ```

2. **Install UVPackMaster:**
   Follow the instructions on the [UVPackMaster website](https://uvpackmaster.com/) to obtain the Linux distribution. Download the ZIP file and place it in the `extern/uvpackmaster` folder.

3. **Install the add-on:**
   We provide a script to install the add-on:

   ```bash
   blenderproc run pack/install_uvp.py
   ```

### Usage

To pack UVs with UVPackMaster, use the same command as the default packing method, changing the `--pack_method` flag to `uvpackmaster`:

```bash
python demo/partuv_demo.py --mesh_path {input_mesh_path} --pack_method uvpackmaster --save_visuals
```

## Multi-Atlas Packing 🚧

---

# 📊 Benchmarking 🚧

---


# 🧱 Building from Source

Please refer to [build.md](doc/build.md) for detailed build instructions.

---

# 🐛 Known Issues

### Handling of non-2-manifold meshes

The ABF assumes the mesh is 2-manifold (each edge is incident to at most two faces). This is currently handled in the preprocessing step, by splitting vertices on non-manifold edges. However, this may create split faces which could result in single-face UV charts. We are working on a better solution to handle this.


# 🔧 Common Problems

Below are common issues and their solutions:

#### 1. Problem with `cuda crt/math_functions.h`

Modify `math_functions.h` according to the fix described at:
[https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591/3](https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591/3)

#### 2. Floating-Point Error

Disable PAMO when running the pipeline on CPU machines.

#### 3. ImportError: `GLIBCXX_3.4.32` not found

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python demo/partuv_demo.py
```

#### 4. (Build) `nvcc fatal: Unsupported gpu architecture 'compute_120'`

Remove `compute_120` from the `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

---



---

# 🍀 Acknowledgement

We acknowledge the following repositories for their contributions and code:

* [PartField](https://github.com/nv-tlabs/PartField)
* [OpenABF](https://github.com/educelab/OpenABF)
* [MeshSimplificationForUnfolding](https://git.ista.ac.at/mbhargav/mesh-simplification-for-unfolding)
* [LSCM](https://github.com/icemiliang/lscm)
* [PAMO](https://github.com/SarahWeiii/pamo)

and all the libraries in the `extern/` folder.

---

# 🎓 BibTeX

If this repository helps your research or project, please consider citing our work:

```bibtex
@inproceedings{wang2025partuv,
  title     = {PartUV: Part-Based UV Unwrapping of 3D Meshes},
  author    = {Wang, Zhaoning and Wei, Xinyue and Shi, Ruoxi and Zhang, Xiaoshuai and Su, Hao and Liu, Minghua},
  booktitle = {ACM SIGGRAPH Asia Conference and Exhibition on Computer Graphics and Interactive Techniques},
  year      = {2025}
}
```
