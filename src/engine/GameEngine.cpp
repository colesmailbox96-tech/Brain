#include "GameEngine.h"
#include "ai/behavior/BehaviorTreeBrain.h"
#include "ai/neural/NeuralBrain.h"
#include <SDL2/SDL.h>
#include <iostream>
#include <sstream>

namespace pw {

GameEngine::GameEngine() : rng(std::random_device{}()) {}

GameEngine::~GameEngine() = default;

void GameEngine::init() {
    // Create world
    world = std::make_unique<World>(42);
    
    // Spawn NPCs
    std::uniform_real_distribution<float> xDist(10.0f, WORLD_WIDTH - 10.0f);
    std::uniform_real_distribution<float> yDist(10.0f, WORLD_HEIGHT - 10.0f);
    
    int neuralCount = 0;
    int behaviorTreeCount = 0;
    
    for (int i = 0; i < 15; i++) {
        Vec2 pos(xDist(rng), yDist(rng));
        
        // Find walkable position
        for (int attempt = 0; attempt < 20; attempt++) {
            if (world->isWalkable(static_cast<int>(pos.x), static_cast<int>(pos.y))) {
                break;
            }
            pos = Vec2(xDist(rng), yDist(rng));
        }
        
        npcs.emplace_back(i, pos);
        
        // Alternate between neural and behavior tree brains (50/50 split)
        if (i % 2 == 0) {
            // Neural brain
            auto neuralBrain = std::make_unique<NeuralBrain>(i, "models/npc_brain.onnx");
            npcs.back().setBrain(std::move(neuralBrain));
            neuralCount++;
        } else {
            // Behavior tree brain (default)
            behaviorTreeCount++;
        }
    }
    
    // Initialize data logger
    dataLogger = std::make_unique<DataLogger>();
    
    std::cout << "Game initialized with " << npcs.size() << " NPCs:" << std::endl;
    std::cout << "  - " << neuralCount << " Neural Brains" << std::endl;
    std::cout << "  - " << behaviorTreeCount << " Behavior Tree Brains" << std::endl;
}

void GameEngine::run() {
    init();
    
    // Create window and rendering systems
    window = std::make_unique<Window>("Pixel World Simulator", 1280, 720);
    window->setVirtualResolution(VIRTUAL_WIDTH, VIRTUAL_HEIGHT);
    
    renderer = std::make_unique<Renderer>(window->getRenderer());
    camera = std::make_unique<Camera>(VIRTUAL_WIDTH, VIRTUAL_HEIGHT);
    debugOverlay = std::make_unique<DebugOverlay>(*renderer);
    input = std::make_unique<InputManager>();
    
    // Center camera on world
    camera->setPosition(Vec2(WORLD_WIDTH / 2.0f, WORLD_HEIGHT / 2.0f));
    camera->setZoom(2.0f);
    
    Uint64 lastTime = SDL_GetPerformanceCounter();
    const Uint64 perfFreq = SDL_GetPerformanceFrequency();
    
    while (running && window->isOpen()) {
        // Calculate delta time
        Uint64 currentTime = SDL_GetPerformanceCounter();
        float dt = static_cast<float>(currentTime - lastTime) / perfFreq;
        lastTime = currentTime;
        
        // Cap dt to avoid spiral of death
        if (dt > 0.25f) dt = 0.25f;
        
        // Process input
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            input->processEvent(event);
        }
        
        handleInput();
        input->update();
        
        // Fixed timestep update
        accumulator += dt;
        while (accumulator >= FIXED_TIMESTEP) {
            update(FIXED_TIMESTEP);
            accumulator -= FIXED_TIMESTEP;
            currentTick++;
        }
        
        // Render
        render();
    }
    
    dataLogger->flush();
    std::cout << "Simulation ended at tick " << currentTick << std::endl;
}

void GameEngine::runHeadless(int ticks) {
    init();
    
    for (int i = 0; i < ticks; i++) {
        update(FIXED_TIMESTEP);
        currentTick++;
        
        if (i % 1000 == 0) {
            std::cout << "Tick: " << currentTick << std::endl;
        }
    }
    
    dataLogger->flush();
    std::cout << "Headless simulation completed: " << currentTick << " ticks" << std::endl;
}

void GameEngine::handleInput() {
    // Camera movement
    float cameraSpeed = 50.0f * FIXED_TIMESTEP;
    Vec2 cameraMove(0, 0);
    
    if (input->isActionPressed(InputAction::MoveUp)) cameraMove.y -= cameraSpeed;
    if (input->isActionPressed(InputAction::MoveDown)) cameraMove.y += cameraSpeed;
    if (input->isActionPressed(InputAction::MoveLeft)) cameraMove.x -= cameraSpeed;
    if (input->isActionPressed(InputAction::MoveRight)) cameraMove.x += cameraSpeed;
    
    camera->move(cameraMove);
    
    // Zoom
    if (input->isActionJustPressed(InputAction::ZoomIn)) {
        camera->setZoom(camera->getZoom() * 1.2f);
    }
    if (input->isActionJustPressed(InputAction::ZoomOut)) {
        camera->setZoom(camera->getZoom() / 1.2f);
    }
    
    // Toggle debug
    if (input->isActionJustPressed(InputAction::ToggleDebug)) {
        showDebug = !showDebug;
    }
    
    // Cycle through NPCs for debug
    if (input->isActionJustPressed(InputAction::CycleNPC) && !npcs.empty()) {
        selectedNPCIndex = (selectedNPCIndex + 1) % npcs.size();
    }
}

void GameEngine::update(float dt) {
    // Update world
    world->update(dt);
    
    // Update NPCs
    for (auto& npc : npcs) {
        // Gather perception
        Perception perception = npc.gatherPerception(*world, npcs);
        
        // Store old needs for delta calculation
        Needs oldNeeds = npc.getNeeds();
        
        // Get decision from brain
        Action action = npc.getBrain()->decide(perception, *world);
        
        // Update NPC
        npc.update(dt, *world, currentTick);
        
        // Calculate outcome
        Outcome outcome;
        Needs newNeeds = npc.getNeeds();
        outcome.needsDeltas["hunger"] = newNeeds.hunger - oldNeeds.hunger;
        outcome.needsDeltas["energy"] = newNeeds.energy - oldNeeds.energy;
        outcome.needsDeltas["social"] = newNeeds.social - oldNeeds.social;
        outcome.needsDeltas["curiosity"] = newNeeds.curiosity - oldNeeds.curiosity;
        outcome.needsDeltas["safety"] = newNeeds.safety - oldNeeds.safety;
        outcome.event = action.toString();
        
        // Log decision
        dataLogger->logDecision(currentTick, npc.getId(), perception, action, outcome);
        
        // Notify brain of outcome
        npc.getBrain()->onOutcome(outcome);
    }
    
    // Log events (NPCs meeting, etc.)
    for (size_t i = 0; i < npcs.size(); i++) {
        for (size_t j = i + 1; j < npcs.size(); j++) {
            float dist = npcs[i].getPosition().distance(npcs[j].getPosition());
            if (dist < 2.0f) {
                json eventData = {
                    {"npc1", npcs[i].getId()},
                    {"npc2", npcs[j].getId()},
                    {"distance", dist}
                };
                dataLogger->logEvent(currentTick, "npc_met", eventData);
            }
        }
    }
}

void GameEngine::render() {
    window->applyVirtualScale();
    window->clear(Color(0, 0, 0));
    
    renderWorld();
    renderNPCs();
    renderWeather();
    
    if (showDebug) {
        renderDebugOverlay();
    }
    
    window->present();
}

void GameEngine::renderWorld() {
    Vec2 camPos = camera->getPosition();
    float zoom = camera->getZoom();
    
    int startX = std::max(0, static_cast<int>(camPos.x - VIRTUAL_WIDTH / (2.0f * zoom * TILE_SIZE)));
    int endX = std::min(WORLD_WIDTH, static_cast<int>(camPos.x + VIRTUAL_WIDTH / (2.0f * zoom * TILE_SIZE)) + 1);
    int startY = std::max(0, static_cast<int>(camPos.y - VIRTUAL_HEIGHT / (2.0f * zoom * TILE_SIZE)));
    int endY = std::min(WORLD_HEIGHT, static_cast<int>(camPos.y + VIRTUAL_HEIGHT / (2.0f * zoom * TILE_SIZE)) + 1);
    
    Color tint = world->getDayNightTint();
    
    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            const Tile& tile = world->getTile(x, y);
            Color color = tile.getColor().withTint(tint, 0.3f);
            
            Vec2 screenPos = camera->worldToScreen(Vec2(x * TILE_SIZE, y * TILE_SIZE));
            int size = static_cast<int>(TILE_SIZE * zoom);
            
            Rect rect(
                static_cast<int>(screenPos.x),
                static_cast<int>(screenPos.y),
                size,
                size
            );
            
            renderer->drawRect(rect, color, true);
        }
    }
}

void GameEngine::renderNPCs() {
    Color tint = world->getDayNightTint();
    
    for (const auto& npc : npcs) {
        Vec2 worldPos = npc.getPosition();
        Vec2 screenPos = camera->worldToScreen(Vec2(worldPos.x * TILE_SIZE, worldPos.y * TILE_SIZE));
        
        int radius = static_cast<int>(4 * camera->getZoom());
        Color color = npc.getColor().withTint(tint, 0.2f);
        
        renderer->drawCircle(
            static_cast<int>(screenPos.x),
            static_cast<int>(screenPos.y),
            radius,
            color,
            true
        );
    }
}

void GameEngine::renderWeather() {
    if (world->getWeather() == Weather::Rain || world->getWeather() == Weather::Storm) {
        // Simple rain effect - draw falling lines
        std::uniform_real_distribution<float> xDist(0, VIRTUAL_WIDTH);
        std::uniform_real_distribution<float> yDist(0, VIRTUAL_HEIGHT);
        
        Color rainColor(150, 150, 200, 100);
        int rainDrops = (world->getWeather() == Weather::Storm) ? 100 : 50;
        
        for (int i = 0; i < rainDrops; i++) {
            int x = static_cast<int>(xDist(rng));
            int y = static_cast<int>(yDist(rng));
            renderer->drawLine(x, y, x + 2, y + 5, rainColor);
        }
    }
}

void GameEngine::renderDebugOverlay() {
    // Draw simple debug info at top-left
    Color textBg(0, 0, 0, 180);
    
    // Background for text
    renderer->drawRect(Rect(0, 0, 200, 80), textBg, true);
    
    // Would normally draw text here, but we don't have a font system
    // Instead, draw colored bars representing aggregate NPC needs
    
    float avgHunger = 0, avgEnergy = 0, avgSocial = 0;
    for (const auto& npc : npcs) {
        avgHunger += npc.getNeeds().hunger;
        avgEnergy += npc.getNeeds().energy;
        avgSocial += npc.getNeeds().social;
    }
    avgHunger /= npcs.size();
    avgEnergy /= npcs.size();
    avgSocial /= npcs.size();
    
    // Hunger bar (red)
    renderer->drawRect(Rect(10, 10, static_cast<int>(avgHunger * 100), 5), Color(255, 0, 0), true);
    // Energy bar (yellow)
    renderer->drawRect(Rect(10, 20, static_cast<int>(avgEnergy * 100), 5), Color(255, 255, 0), true);
    // Social bar (blue)
    renderer->drawRect(Rect(10, 30, static_cast<int>(avgSocial * 100), 5), Color(0, 150, 255), true);
    
    // Time of day indicator
    int todX = static_cast<int>(world->getTimeOfDay() * 180) + 10;
    renderer->drawRect(Rect(todX, 45, 5, 10), Color(255, 255, 0), true);
    
    // Render detailed NPC debug panel for selected NPC
    if (!npcs.empty() && selectedNPCIndex < static_cast<int>(npcs.size())) {
        const NPC& selectedNPC = npcs[selectedNPCIndex];
        
        // Highlight selected NPC
        Vec2 screenPos = camera->worldToScreen(selectedNPC.getPosition());
        renderer->drawCircle(static_cast<int>(screenPos.x), static_cast<int>(screenPos.y), 
                           12, Color(255, 255, 0), false);
        
        // Render debug panel
        if (debugOverlay) {
            debugOverlay->renderNPCDebug(selectedNPC, VIRTUAL_WIDTH - 410, 10);
        }
    }
}

} // namespace pw
