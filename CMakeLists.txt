cmake_minimum_required(VERSION 3.18)
project(HetKVCache VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC -pthread")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

# Find packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

# CUDA paths
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.8)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
set(WSL_CUDA_LIB_DIR /usr/lib/wsl/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CUDA_INCLUDE_DIRS})

# Link directories
link_directories(${CUDA_LIB_DIR})
link_directories(${WSL_CUDA_LIB_DIR})

# Source files
set(CORE_SOURCES
    src/core/kv_cache_manager.cpp
    src/core/memory_pool.cpp
    src/core/block_allocator.cpp
    src/monitor/access_monitor.cpp
    src/monitor/heat_evaluator.cpp
    src/monitor/access_pattern_analyzer.cpp
    src/migration/migration_engine.cpp
    src/migration/transfer_scheduler.cpp
    src/migration/prefetcher.cpp
    src/allocator/paged_allocator.cpp
    src/allocator/memory_mapper.cpp
    src/utils/logger.cpp
    src/utils/config.cpp
    src/utils/timer.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/memory_transfer.cu
    src/cuda/kernels.cu
    src/cuda/stream_manager.cu
)

# Create shared library
add_library(hetkvcache SHARED ${CORE_SOURCES} ${CUDA_SOURCES})
target_link_libraries(hetkvcache 
    Threads::Threads
    ${CUDA_LIB_DIR}/libcudart.so
    ${WSL_CUDA_LIB_DIR}/libcuda.so
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hetkvcache OpenMP::OpenMP_CXX)
endif()

# Create static library (CUDA code can't be in static library easily, so we compile as C++)
add_library(hetkvcache_static STATIC ${CORE_SOURCES})
target_link_libraries(hetkvcache_static 
    Threads::Threads
    ${CUDA_LIB_DIR}/libcudart.so
    ${WSL_CUDA_LIB_DIR}/libcuda.so
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hetkvcache_static OpenMP::OpenMP_CXX)
endif()

# Header-only interface library
add_library(hetkvcache_headers INTERFACE)
target_include_directories(hetkvcache_headers INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# Test executable
add_executable(hetkvcache_test tests/test_main.cpp)
target_link_libraries(hetkvcache_test hetkvcache)

# Benchmark executable
add_executable(hetkvcache_bench benchmarks/benchmark_main.cpp)
target_link_libraries(hetkvcache_bench hetkvcache)

# Install rules
install(TARGETS hetkvcache hetkvcache_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# Add subdirectories for examples
add_subdirectory(examples)