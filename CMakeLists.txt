cmake_minimum_required(VERSION 3.15)
project(PixelWorldSimulator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find SDL2
find_package(SDL2 REQUIRED)
find_package(SDL2_mixer QUIET)

# nlohmann/json - header-only library
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# ONNX Runtime for neural inference (optional - will use fallback if not available)
# Download ONNX Runtime manually if needed:
# https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/include" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/lib" CACHE PATH "ONNX Runtime library directory")

if(EXISTS ${ONNXRUNTIME_INCLUDE_DIR})
    message(STATUS "ONNX Runtime found at ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(WARNING "ONNX Runtime not found - neural brain will use fallback behavior")
    message(WARNING "To enable neural inference, download ONNX Runtime and extract to external/onnxruntime/")
endif()

# Source files
file(GLOB_RECURSE SOURCES
    src/engine/*.cpp
    src/platform/*.cpp
    src/world/*.cpp
    src/entities/*.cpp
    src/ai/**/*.cpp
    src/rendering/*.cpp
    src/input/*.cpp
    src/data/*.cpp
    src/serialization/*.cpp
)

# Main executable
add_executable(pixel_world_sim
    src/main.cpp
    ${SOURCES}
)

target_include_directories(pixel_world_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SDL2_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(pixel_world_sim PRIVATE
    ${SDL2_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Link ONNX Runtime
if(EXISTS ${ONNXRUNTIME_LIB_DIR})
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)
    if(ONNXRUNTIME_LIB)
        target_link_libraries(pixel_world_sim PRIVATE ${ONNXRUNTIME_LIB})
        target_compile_definitions(pixel_world_sim PRIVATE HAS_ONNX_RUNTIME)
    endif()
endif()

if(SDL2_mixer_FOUND)
    target_link_libraries(pixel_world_sim PRIVATE SDL2_mixer)
    target_compile_definitions(pixel_world_sim PRIVATE HAS_SDL_MIXER)
endif()

# Platform-specific settings
if(WIN32)
    target_link_libraries(pixel_world_sim PRIVATE mingw32)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(pixel_world_sim PRIVATE m pthread)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(pixel_world_sim PRIVATE /W4)
else()
    target_compile_options(pixel_world_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()
