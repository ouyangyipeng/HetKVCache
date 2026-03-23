cmake_minimum_required(VERSION 3.18)

# ============================================================================
# 项目配置
# ============================================================================

# 构建选项
option(USE_ASCEND "Build with Ascend NPU support" ON)
option(USE_CUDA "Build with CUDA support" OFF)
option(USE_CPU_FALLBACK "Build with CPU fallback (no GPU/NPU)" OFF)

# 根据选项选择项目语言
if(USE_CUDA)
    project(HetKVCache VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
    project(HetKVCache VERSION 1.0.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 编译选项
# ============================================================================

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC -pthread")

# ARM架构优化（鲲鹏920）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Detected ARM64 architecture (Kunpeng 920)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+crypto+fp16+dotprod")
endif()

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")
endif()

# ============================================================================
# 查找依赖
# ============================================================================

find_package(Threads REQUIRED)
find_package(OpenMP)

# ============================================================================
# 昇腾CANN配置
# ============================================================================

if(USE_ASCEND)
    message(STATUS "Building with Ascend NPU support")
    
    # CANN路径配置
    set(CANN_ROOT "/usr/local/Ascend/ascend-toolkit/latest" 
         CACHE PATH "Path to CANN installation")
    
    # 检查CANN是否存在
    if(NOT EXISTS ${CANN_ROOT})
        message(WARNING "CANN not found at ${CANN_ROOT}, falling back to CPU mode")
        set(USE_ASCEND OFF)
        set(USE_CPU_FALLBACK ON)
    else()
        message(STATUS "CANN found at ${CANN_ROOT}")
        
        # ACL头文件和库路径
        set(CANN_INCLUDE_DIRS 
            ${CANN_ROOT}/include
            ${CANN_ROOT}/include/acl
        )
        set(CANN_LIB_DIRS 
            ${CANN_ROOT}/lib64
        )
        
        # 添加昇腾源文件
        set(ASCEND_SOURCES
            src/npu/ascend_adapter.cpp
            src/device/device_factory.cpp
        )
        
        include_directories(${CANN_INCLUDE_DIRS})
        link_directories(${CANN_LIB_DIRS})
        
        add_definitions(-DUSE_ASCEND)
    endif()
endif()

# ============================================================================
# CUDA配置
# ============================================================================

if(USE_CUDA AND NOT USE_ASCEND)
    message(STATUS "Building with CUDA support")
    
    find_package(CUDA REQUIRED)
    
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.8)
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    set(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
    set(WSL_CUDA_LIB_DIR /usr/lib/wsl/lib)
    
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIB_DIR})
    link_directories(${WSL_CUDA_LIB_DIR})
    
    set(CUDA_SOURCES
        src/cuda/memory_transfer.cu
        src/cuda/kernels.cu
        src/cuda/stream_manager.cu
    )
    
    add_definitions(-DUSE_CUDA)
endif()

# ============================================================================
# CPU Fallback配置
# ============================================================================

if(USE_CPU_FALLBACK AND NOT USE_ASCEND AND NOT USE_CUDA)
    message(STATUS "Building with CPU fallback (no GPU/NPU)")
    
    set(CPU_SOURCES
        src/device/device_factory.cpp
    )
    
    add_definitions(-DUSE_CPU_FALLBACK)
endif()

# ============================================================================
# 包含目录
# ============================================================================

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# ============================================================================
# 源文件
# ============================================================================

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

# ============================================================================
# 库构建
# ============================================================================

# 根据后端选择源文件
if(USE_ASCEND)
    set(BACKEND_SOURCES ${ASCEND_SOURCES})
    set(BACKEND_LIBS ascendcl)
elseif(USE_CUDA)
    set(BACKEND_SOURCES ${CUDA_SOURCES})
    set(BACKEND_LIBS 
        ${CUDA_LIB_DIR}/libcudart.so
        ${WSL_CUDA_LIB_DIR}/libcuda.so
    )
else()
    set(BACKEND_SOURCES ${CPU_SOURCES})
    set(BACKEND_LIBS "")
endif()

# 共享库
add_library(hetkvcache SHARED ${CORE_SOURCES} ${BACKEND_SOURCES})
target_link_libraries(hetkvcache 
    Threads::Threads
    ${BACKEND_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hetkvcache OpenMP::OpenMP_CXX)
endif()

# 静态库
add_library(hetkvcache_static STATIC ${CORE_SOURCES} ${BACKEND_SOURCES})
target_link_libraries(hetkvcache_static 
    Threads::Threads
    ${BACKEND_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hetkvcache_static OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# 可执行文件
# ============================================================================

# 测试程序
add_executable(hetkvcache_test tests/test_main.cpp)
target_link_libraries(hetkvcache_test hetkvcache)

# 基准测试程序
add_executable(hetkvcache_bench benchmarks/benchmark_main.cpp)
target_link_libraries(hetkvcache_bench hetkvcache)

# ============================================================================
# 安装规则
# ============================================================================

install(TARGETS hetkvcache hetkvcache_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# ============================================================================
# 示例子目录
# ============================================================================

add_subdirectory(examples)

# ============================================================================
# 打印配置摘要
# ============================================================================

message(STATUS "")
message(STATUS "=== HetKVCache Build Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "Backend Options:")
message(STATUS "  USE_ASCEND:      ${USE_ASCEND}")
message(STATUS "  USE_CUDA:        ${USE_CUDA}")
message(STATUS "  USE_CPU_FALLBACK: ${USE_CPU_FALLBACK}")
message(STATUS "")

if(USE_ASCEND)
    message(STATUS "CANN Configuration:")
    message(STATUS "  CANN_ROOT:       ${CANN_ROOT}")
    message(STATUS "  CANN_INCLUDES:   ${CANN_INCLUDE_DIRS}")
    message(STATUS "  CANN_LIBS:       ${CANN_LIB_DIRS}")
    message(STATUS "")
endif()

if(USE_CUDA)
    message(STATUS "CUDA Configuration:")
    message(STATUS "  CUDA_ROOT:       ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "  CUDA_INCLUDES:   ${CUDA_INCLUDE_DIRS}")
    message(STATUS "  CUDA_LIBS:       ${CUDA_LIB_DIR}")
    message(STATUS "")
endif()

message(STATUS "======================================")