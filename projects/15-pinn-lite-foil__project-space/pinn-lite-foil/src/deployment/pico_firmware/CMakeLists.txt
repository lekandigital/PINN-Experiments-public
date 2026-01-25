# PINN-Lite-Foil Raspberry Pi Pico Firmware
# AoA (Angle of Attack) Streaming

cmake_minimum_required(VERSION 3.13)

# Include Pico SDK
set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(aoa_stream C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize Pico SDK
pico_sdk_init()

# Main executable
add_executable(aoa_stream
    aoa_stream.c
)

# Link Pico SDK libraries
target_link_libraries(aoa_stream
    pico_stdlib
    hardware_i2c
    hardware_gpio
    hardware_uart
)

# Enable USB output
pico_enable_stdio_usb(aoa_stream 1)
pico_enable_stdio_uart(aoa_stream 1)

# Generate UF2 file for flashing
pico_add_extra_outputs(aoa_stream)

# Print build info
message(STATUS "")
message(STATUS "=== Pico Firmware Configuration ===")
message(STATUS "  Pico SDK Path: ${PICO_SDK_PATH}")
message(STATUS "  Target: aoa_stream")
message(STATUS "===================================")
message(STATUS "")
