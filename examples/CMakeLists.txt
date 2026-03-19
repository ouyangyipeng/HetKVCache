# HetKVCache Examples CMakeLists.txt

# Basic usage example
add_executable(basic_example
    basic_usage/basic_example.cpp
)
target_link_libraries(basic_example PRIVATE hetkvcache)
target_include_directories(basic_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Set output directory
set_target_properties(basic_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)