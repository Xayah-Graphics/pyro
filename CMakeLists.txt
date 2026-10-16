# ============================================================================
# Project.
# ============================================================================
cmake_minimum_required(VERSION 4.4)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" PYRO_VERSION LIMIT_COUNT 1)
project(PYRO VERSION "${PYRO_VERSION}" LANGUAGES CUDA)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_ARCHITECTURES 120a-real)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Dependencies
# ============================================================================
include(cmake/cuda.cmake)

# ============================================================================
# Geometry
# ============================================================================
add_library(geometry STATIC pyro/geometry/geometry.cu pyro/geometry/bvh.cu)
target_link_libraries(geometry PUBLIC CUDA::cudart)
target_include_directories(geometry PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pyro")

# ============================================================================
# Solver
# ============================================================================
foreach(backend IN ITEMS sparse dense lfm)
    add_library(pyro_${backend} STATIC
        pyro/solver/${backend}/solver.cu
        pyro/solver/${backend}/grid.cu
        pyro/solver/${backend}/advection.cu
        pyro/solver/${backend}/pressure.cu
        pyro/solver/${backend}/statistics.cu
    )
    target_link_libraries(pyro_${backend} PUBLIC geometry)
endforeach()
target_sources(pyro_dense PRIVATE pyro/solver/dense/forces.cu)
target_sources(pyro_sparse PRIVATE pyro/solver/sparse/forces.cu pyro/solver/sparse/transfer.cu)
target_sources(pyro_lfm PRIVATE pyro/solver/lfm/flow_map.cu)

# ============================================================================
# Executables
# ============================================================================
add_executable(pyro_benchmark pyro/benchmark/benchmark.cu)
add_executable(pyro_stream pyro/io/stream.cu)
add_executable(pyro_dense_benchmark pyro/benchmark/dense.cu)
target_link_libraries(pyro_dense_benchmark PRIVATE pyro_dense)
foreach(target IN ITEMS pyro_benchmark pyro_stream)
    target_link_libraries(${target} PRIVATE pyro_sparse pyro_dense pyro_lfm)
endforeach()

option(PYRO_CUDA_LINEINFO "Emit CUDA source locations for profiling" OFF)
option(PYRO_CUDA_RESOURCE_REPORT "Report per-kernel CUDA register and memory usage" OFF)
foreach(target IN ITEMS geometry pyro_sparse pyro_dense pyro_lfm pyro_benchmark pyro_stream pyro_dense_benchmark)
    if(PYRO_CUDA_LINEINFO)
        target_compile_options(${target} PRIVATE -lineinfo)
    endif()
    if(PYRO_CUDA_RESOURCE_REPORT)
        target_compile_options(${target} PRIVATE -Xptxas=-v)
    endif()
endforeach()

# ============================================================================
# Blender (Windows).
# ============================================================================
if(WIN32)
    include(cmake/blender.cmake)
endif()
