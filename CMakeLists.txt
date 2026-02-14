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

# ONNX Runtime for neural inference
FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
    URL_HASH SHA256=9d6c7921e3131c3a8b8f4b1e3f0b1c7c3d3d3e3f3e3f3e3f3e3f3e3f3e3f3e3f
)
FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
    FetchContent_Populate(onnxruntime)
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
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
