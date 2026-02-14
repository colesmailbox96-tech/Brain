#pragma once

#include <cstdint>

namespace pw {

// Core types
using EntityId = uint32_t;
using Tick = uint64_t;

// Constants
constexpr int VIRTUAL_WIDTH = 480;
constexpr int VIRTUAL_HEIGHT = 270;
constexpr float FIXED_TIMESTEP = 1.0f / 60.0f; // 60 Hz simulation
constexpr int MAX_FRAME_SKIP = 5;

// World constants
constexpr int WORLD_WIDTH = 200;
constexpr int WORLD_HEIGHT = 150;
constexpr int TILE_SIZE = 8;

} // namespace pw
