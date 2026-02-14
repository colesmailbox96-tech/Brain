#pragma once

#include "engine/Math.h"
#include <cstdint>

namespace pw {

enum class TileType : uint8_t {
    Water,
    Sand,
    Grass,
    Dirt,
    Stone,
    Tree,
    BerryBush,
    Cave,
    Shelter // Player-built
};

struct Tile {
    TileType type = TileType::Grass;
    bool walkable = true;
    bool hasFood = false;
    int foodAmount = 0;
    
    Color getColor() const;
    bool isResource() const;
};

} // namespace pw
