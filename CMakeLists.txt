# ============================================================================
# Project.
# ============================================================================
cmake_minimum_required(VERSION 4.4)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" PYRO_VERSION LIMIT_COUNT 1)
project(PYRO VERSION "${PYRO_VERSION}" LANGUAGES C CXX CUDA)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)

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
# Solver
# ============================================================================
add_library(pyro_geometry STATIC solver/geometry.cpp solver/geometry.cu)
target_sources(pyro_geometry PUBLIC FILE_SET CXX_MODULES FILES solver/types.ixx solver/geometry.ixx)
target_link_libraries(pyro_geometry PUBLIC CUDA::cudart)
foreach(backend IN ITEMS sparse dense)
    add_library(pyro_${backend} STATIC
        solver/${backend}/solver.cpp
        solver/${backend}/grid.cu
        solver/${backend}/advection.cu
        solver/${backend}/pressure.cu
    )
    target_sources(pyro_${backend} PUBLIC FILE_SET CXX_MODULES FILES solver/${backend}/solver.ixx)
    target_link_libraries(pyro_${backend} PUBLIC pyro_geometry)
endforeach()
target_sources(pyro_sparse PRIVATE solver/sparse/transfer.cu)
foreach(target IN ITEMS pyro_geometry pyro_sparse pyro_dense)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>")
    target_compile_options(${target} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>")
endforeach()

# ============================================================================
# Executables
# ============================================================================
add_executable(pyro main.cpp io/stream.cpp)
target_sources(pyro PRIVATE FILE_SET CXX_MODULES FILES io/stream.ixx)
target_link_libraries(pyro PRIVATE pyro_sparse pyro_dense)

# ============================================================================
# Blender (Windows).
# ============================================================================
if(WIN32)
    include(cmake/blender.cmake)
endif()
