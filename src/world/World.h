#pragma once

#include "Tile.h"
#include "SimplexNoise.h"
#include "engine/Types.h"
#include <vector>
#include <memory>

namespace pw {

enum class Weather {
    Clear,
    Rain,
    Storm
};

class World {
public:
    World(uint32_t seed = 42);
    
    void update(float dt);
    
    Tile& getTile(int x, int y);
    const Tile& getTile(int x, int y) const;
    bool isWalkable(int x, int y) const;
    
    float getTimeOfDay() const { return timeOfDay; }
    Weather getWeather() const { return currentWeather; }
    Color getDayNightTint() const;
    
    int getWidth() const { return WORLD_WIDTH; }
    int getHeight() const { return WORLD_HEIGHT; }

private:
    void generateTerrain();
    void updateDayNight(float dt);
    void updateWeather(float dt);
    
    std::vector<Tile> tiles;
    SimplexNoise noise;
    
    float timeOfDay = 0.0f; // 0.0 = midnight, 0.5 = noon, 1.0 = midnight
    float dayNightSpeed = 0.02f; // Full day cycle takes ~50 seconds
    
    Weather currentWeather = Weather::Clear;
    float weatherTimer = 0.0f;
    float weatherDuration = 0.0f;
};

} // namespace pw
