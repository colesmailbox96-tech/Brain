#pragma once

#include "Types.h"
#include "world/World.h"
#include "entities/NPC.h"
#include "data/DataLogger.h"
#include "platform/Window.h"
#include "rendering/Renderer.h"
#include "rendering/Camera.h"
#include "rendering/DebugOverlay.h"
#include "input/InputManager.h"
#include <vector>
#include <memory>
#include <random>

namespace pw {

class GameEngine {
public:
    GameEngine();
    ~GameEngine();
    
    void run();
    void runHeadless(int ticks);

private:
    void init();
    void update(float dt);
    void render();
    void handleInput();
    
    void renderWorld();
    void renderNPCs();
    void renderWeather();
    void renderDebugOverlay();
    
    std::unique_ptr<Window> window;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<Camera> camera;
    std::unique_ptr<DebugOverlay> debugOverlay;
    std::unique_ptr<InputManager> input;
    
    std::unique_ptr<World> world;
    std::vector<NPC> npcs;
    std::unique_ptr<DataLogger> dataLogger;
    
    Tick currentTick = 0;
    float accumulator = 0.0f;
    bool showDebug = false;
    int selectedNPCIndex = 0;
    bool running = true;
    
    std::mt19937 rng;
};

} // namespace pw
