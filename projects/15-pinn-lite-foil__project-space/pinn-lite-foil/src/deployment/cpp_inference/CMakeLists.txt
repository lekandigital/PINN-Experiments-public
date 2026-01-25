# PINN Inference C++ Wrapper
# Requires ONNX Runtime

cmake_minimum_required(VERSION 3.14)
project(pinn_inference VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find ONNX Runtime
if(NOT DEFINED ONNXRUNTIME_ROOT)
    # Try to find in common locations
    if(EXISTS "/usr/local/onnxruntime")
        set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
    elseif(EXISTS "$ENV{HOME}/onnxruntime")
        set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime")
    elseif(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    else()
        message(FATAL_ERROR "ONNXRUNTIME_ROOT not set. Please set it to the ONNX Runtime installation directory.")
    endif()
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

# ONNX Runtime include and library paths
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not find onnxruntime library in ${ONNXRUNTIME_LIB_DIR}")
endif()

message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# Check for CUDA support
option(USE_CUDA "Use CUDA execution provider" OFF)

if(USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA not found, disabling CUDA support")
        set(USE_CUDA OFF)
    endif()
endif()

# Main executable
add_executable(pinn_inference pinn_inference.cpp)

target_include_directories(pinn_inference PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(pinn_inference PRIVATE
    ${ONNXRUNTIME_LIB}
)

# RPATH for finding shared libraries at runtime
set_target_properties(pinn_inference PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Installation
install(TARGETS pinn_inference DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "=== PINN Inference Configuration ===")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime:    ${ONNXRUNTIME_ROOT}")
message(STATUS "  CUDA Support:    ${USE_CUDA}")
message(STATUS "=====================================")
message(STATUS "")
