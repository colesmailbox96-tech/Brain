#include "World.h"
#include <random>
#include <cmath>

namespace pw {

World::World(uint32_t seed) : noise(seed) {
    tiles.resize(WORLD_WIDTH * WORLD_HEIGHT);
    generateTerrain();
    
    // Initialize weather
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(10.0f, 30.0f);
    weatherDuration = dist(rng);
}

void World::generateTerrain() {
    for (int y = 0; y < WORLD_HEIGHT; y++) {
        for (int x = 0; x < WORLD_WIDTH; x++) {
            float elevation = noise.octaveNoise(x * 0.05f, y * 0.05f, 4, 0.5f);
            float moisture = noise.octaveNoise(x * 0.03f + 100, y * 0.03f + 100, 3, 0.5f);
            float detail = noise.octaveNoise(x * 0.2f, y * 0.2f, 2, 0.4f);
            
            Tile& tile = getTile(x, y);
            
            // Determine tile type based on noise
            if (elevation < -0.3f) {
                tile.type = TileType::Water;
                tile.walkable = false;
            } else if (elevation < -0.15f) {
                tile.type = TileType::Sand;
            } else if (elevation > 0.5f) {
                if (moisture < -0.2f) {
                    tile.type = TileType::Stone;
                } else if (detail > 0.3f && moisture > 0.0f) {
                    tile.type = TileType::Cave;
                } else {
                    tile.type = TileType::Stone;
                }
            } else {
                // Mid elevation - grass, dirt, vegetation
                if (moisture > 0.3f && detail > 0.4f) {
                    tile.type = TileType::Tree;
                    tile.walkable = false;
                } else if (moisture > 0.0f && detail > 0.5f) {
                    tile.type = TileType::BerryBush;
                    tile.hasFood = true;
                    tile.foodAmount = 5;
                } else if (moisture < -0.2f) {
                    tile.type = TileType::Dirt;
                } else {
                    tile.type = TileType::Grass;
                }
            }
        }
    }
}

Tile& World::getTile(int x, int y) {
    if (x < 0 || x >= WORLD_WIDTH || y < 0 || y >= WORLD_HEIGHT) {
        static Tile invalid;
        invalid.walkable = false;
        return invalid;
    }
    return tiles[y * WORLD_WIDTH + x];
}

const Tile& World::getTile(int x, int y) const {
    if (x < 0 || x >= WORLD_WIDTH || y < 0 || y >= WORLD_HEIGHT) {
        static Tile invalid;
        invalid.walkable = false;
        return invalid;
    }
    return tiles[y * WORLD_WIDTH + x];
}

bool World::isWalkable(int x, int y) const {
    return getTile(x, y).walkable;
}

void World::update(float dt) {
    updateDayNight(dt);
    updateWeather(dt);
}

void World::updateDayNight(float dt) {
    timeOfDay += dt * dayNightSpeed;
    if (timeOfDay >= 1.0f) {
        timeOfDay -= 1.0f;
    }
}

void World::updateWeather(float dt) {
    weatherTimer += dt;
    
    if (weatherTimer >= weatherDuration) {
        weatherTimer = 0.0f;
        
        // Change weather
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> chanceDist(0.0f, 1.0f);
        std::uniform_real_distribution<float> durationDist(10.0f, 30.0f);
        
        float chance = chanceDist(gen);
        if (currentWeather == Weather::Clear) {
            if (chance < 0.3f) {
                currentWeather = Weather::Rain;
            } else if (chance < 0.4f) {
                currentWeather = Weather::Storm;
            }
        } else {
            if (chance < 0.6f) {
                currentWeather = Weather::Clear;
            }
        }
        
        weatherDuration = durationDist(gen);
    }
}

Color World::getDayNightTint() const {
    // Calculate lighting based on time of day
    float lightLevel = 1.0f;
    
    if (timeOfDay < 0.25f) {
        // Night -> Dawn (0.0 - 0.25)
        lightLevel = 0.3f + (timeOfDay / 0.25f) * 0.7f;
    } else if (timeOfDay < 0.75f) {
        // Day (0.25 - 0.75)
        lightLevel = 1.0f;
    } else {
        // Dusk -> Night (0.75 - 1.0)
        lightLevel = 1.0f - ((timeOfDay - 0.75f) / 0.25f) * 0.7f;
    }
    
    // Night has a blue tint
    if (timeOfDay < 0.25f || timeOfDay > 0.75f) {
        float nightStrength = 1.0f - lightLevel;
        return Color(
            static_cast<uint8_t>(255 * lightLevel),
            static_cast<uint8_t>(255 * lightLevel),
            static_cast<uint8_t>(255 * (lightLevel + nightStrength * 0.3f))
        );
    }
    
    return Color(
        static_cast<uint8_t>(255 * lightLevel),
        static_cast<uint8_t>(255 * lightLevel),
        static_cast<uint8_t>(255 * lightLevel)
    );
}

} // namespace pw
