#include "Tile.h"

namespace pw {

Color Tile::getColor() const {
    switch (type) {
        case TileType::Water:
            return Color(50, 100, 200);
        case TileType::Sand:
            return Color(220, 200, 150);
        case TileType::Grass:
            return Color(80, 180, 80);
        case TileType::Dirt:
            return Color(140, 100, 60);
        case TileType::Stone:
            return Color(120, 120, 130);
        case TileType::Tree:
            return Color(60, 120, 40);
        case TileType::BerryBush:
            return Color(100, 150, 60);
        case TileType::Cave:
            return Color(60, 60, 70);
        case TileType::Shelter:
            return Color(150, 120, 80);
        default:
            return Color(255, 255, 255);
    }
}

bool Tile::isResource() const {
    return type == TileType::BerryBush || type == TileType::Tree;
}

} // namespace pw
