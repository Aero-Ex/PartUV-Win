cmake_minimum_required(VERSION 3.18) # Increased to 3.18 for better CUDA support

project(partuv LANGUAGES CXX CUDA)

# --- 1. Architecture and Compiler Setup ---
set(CMAKE_CUDA_ARCHITECTURES 80 89 90)

# Only look for manual path if NOT on Windows (MSVC finds it via Toolset)
if(NOT MSVC)
  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER NAMES nvcc)
    if(NOT CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
      message(WARNING "Could not find 'nvcc' in PATH. Defaulting to /usr/local/cuda-12.1/bin/nvcc.")
    endif()
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")

# --- 2. Platform Specific Flags ---
if(MSVC)
    # Windows / Visual Studio Flags (only for C/CXX, not CUDA)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    )

    # Fix compatibility issues (Math constants, Linux keywords, std::byte conflicts)
    add_compile_definitions(
        _USE_MATH_DEFINES
        NOMINMAX
        M_PI=3.14159265358979323846
        _HAS_STD_BYTE=0
    )

    # Suppress some common NVCC warnings on Windows
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd4819\"") 
else()
    # Linux / GCC / Clang Flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -mfpmath=sse -Ofast -fno-omit-frame-pointer -march=native")
endif()

option(ENABLE_PROFILING "Enable or disable profiling timers" ON)

# --- 3. Dependencies ---
set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(LIBIGL_INCLUDE_DIR ${EXTERNAL_DIR}/libigl/include)
set(EIGEN_INCLUDE_DIR ${EXTERNAL_DIR}/eigen-3.4.0)
set(OPENABF_INCLUDE_DIR ${EXTERNAL_DIR}/OpenABF/include)
set(STB_INCLUDE_DIR ${EXTERNAL_DIR}/stb)
set(NLOHMANN_JSON_INCLUDE_DIR ${EXTERNAL_DIR}/json/include)

include_directories(
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
    ${OPENABF_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

find_package(CGAL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED) 
find_package(pybind11 CONFIG REQUIRED)

# Handle Easy Profiler (Windows DLL vs Linux SO)
find_package(easy_profiler REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/easy_profiler/lib/cmake/easy_profiler") 
get_target_property(_EP_SO easy_profiler IMPORTED_LOCATION_RELEASE)
if(NOT _EP_SO)
  get_target_property(_EP_SO easy_profiler IMPORTED_LOCATION)
endif()

# On Windows, we often need to install the .dll next to the pyd
if(WIN32)
    install(FILES "${_EP_SO}" DESTINATION partuv)
else()
    install(FILES "${_EP_SO}" DESTINATION partuv/.libs)
endif()


# --- 4. Subdirectories ---
add_subdirectory(${SRC_DIR}/lscm)
target_compile_options(lscm_lib PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fPIC>) # fPIC not needed on Windows

add_subdirectory(${SRC_DIR}/triangle_intersection)
target_compile_options(triangle_intersection PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fPIC>)

# --- 5. Source Collection ---
option(USE_ALL_SRC_FILES "Use all .cpp files in src directory" ON)

if(USE_ALL_SRC_FILES)
    file(GLOB SOURCES "${SRC_DIR}/*.cpp")
    file(GLOB CUDA_SOURCES "${SRC_DIR}/*.cu")
    set(BINDINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp")
    list(APPEND SOURCES ${CUDA_SOURCES})
    list(APPEND SOURCES ${BINDINGS_SOURCES})
else()
    set(SOURCES "${SRC_DIR}/Distortion.cpp" "${SRC_DIR}/IO.cpp")
endif()

if(NOT WIN32)
    set(ENV{OMP_NUM_THREADS} "8")
endif()

# --- 6. Python Module ---
pybind11_add_module(_core bindings.cpp ${SOURCES})

set_target_properties(_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(_core PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-extended-lambda
    --expt-relaxed-constexpr
    --default-stream=per-thread 
  >
)

target_link_libraries(_core
    PUBLIC
        easy_profiler
    PRIVATE
        lscm_lib
        triangle_intersection
        OpenMP::OpenMP_CXX
        yaml-cpp::yaml-cpp
        TBB::tbb
        CUDA::cudart
)

target_include_directories(_core PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
    ${OPENABF_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# --- 7. Installation ---
install(TARGETS _core DESTINATION partuv)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/partuv/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/partuv/preprocess.py
  DESTINATION partuv)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/preprocess_utils/
  DESTINATION partuv/preprocess_utils
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.yaml"
    PATTERN "*.yml"
    PATTERN "*.json"
    PATTERN "*.txt"
    PATTERN "*.pt"
    PATTERN "*.bin"
)

if(NOT WIN32)
    set_target_properties(_core PROPERTIES
      BUILD_RPATH  "\$ORIGIN;\$ORIGIN/.libs"
      INSTALL_RPATH "\$ORIGIN;\$ORIGIN/.libs"
    )
endif()