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
)

target_link_libraries(pixel_world_sim PRIVATE
    ${SDL2_LIBRARIES}
    nlohmann_json::nlohmann_json
)

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
