#include "NPC.h"
#include "world/World.h"
#include "ai/behavior/BehaviorTreeBrain.h"
#include <random>
#include <algorithm>

namespace pw {

NPC::NPC(EntityId id, Vec2 position) : id(id), position(position) {
    // Random color for visual distinction
    std::mt19937 rng(id);
    std::uniform_int_distribution<int> colorDist(100, 255);
    color = Color(colorDist(rng), colorDist(rng), colorDist(rng));
    
    // Initialize with behavior tree brain
    brain = std::make_unique<BehaviorTreeBrain>(id);
}

void NPC::setBrain(std::unique_ptr<IBrain> newBrain) {
    brain = std::move(newBrain);
}

void NPC::update(float dt, World& world, Tick currentTick) {
    updateNeeds(dt);
    updateMood();
    
    // Make decision
    if (brain) {
        Perception perception = gatherPerception(world, {});
        currentAction = brain->decide(perception, world);
    }
    
    executeAction(dt, world);
}

void NPC::updateNeeds(float dt) {
    needs.update(dt);
}

void NPC::updateMood() {
    // Determine mood based on needs
    float avgNeed = (needs.hunger + needs.energy + needs.social) / 3.0f;
    
    if (avgNeed < 0.3f) {
        mood = Mood::Happy;
    } else if (avgNeed < 0.5f) {
        mood = Mood::Neutral;
    } else if (avgNeed < 0.7f) {
        mood = Mood::Anxious;
    } else {
        mood = Mood::Sad;
    }
    
    if (needs.curiosity > 0.7f) {
        mood = Mood::Excited;
    }
}

void NPC::executeAction(float dt, World& world) {
    switch (currentAction.type) {
        case ActionType::Move:
        case ActionType::Explore:
        case ActionType::SeekShelter:
            moveTowards(currentAction.targetPosition, dt);
            break;
            
        case ActionType::Eat: {
            int x = static_cast<int>(position.x);
            int y = static_cast<int>(position.y);
            Tile& tile = world.getTile(x, y);
            if (tile.hasFood && tile.foodAmount > 0) {
                tile.foodAmount--;
                needs.hunger = std::max(0.0f, needs.hunger - 0.3f);
                if (tile.foodAmount <= 0) {
                    tile.hasFood = false;
                }
            }
            break;
        }
        
        case ActionType::Rest:
            needs.energy = std::max(0.0f, needs.energy - dt * 0.2f);
            break;
            
        case ActionType::Socialize:
            needs.social = std::max(0.0f, needs.social - dt * 0.1f);
            break;
            
        case ActionType::Idle:
        default:
            break;
    }
}

void NPC::moveTowards(Vec2 target, float dt) {
    Vec2 direction = (target - position).normalized();
    float distance = position.distance(target);
    
    // Adjust speed based on mood
    float currentSpeed = speed;
    switch (mood) {
        case Mood::Happy:
        case Mood::Excited:
            currentSpeed = speed * 1.2f;
            break;
        case Mood::Sad:
        case Mood::Anxious:
            currentSpeed = speed * 0.8f;
            break;
        default:
            break;
    }
    
    float moveAmount = currentSpeed * dt;
    if (distance <= moveAmount) {
        position = target;
    } else {
        position = position + direction * moveAmount;
    }
    
    // Clamp to world bounds
    position.x = std::max(0.0f, std::min(static_cast<float>(WORLD_WIDTH - 1), position.x));
    position.y = std::max(0.0f, std::min(static_cast<float>(WORLD_HEIGHT - 1), position.y));
}

Perception NPC::gatherPerception(const World& world, const std::vector<NPC>& allNPCs) const {
    Perception p;
    p.position = position;
    p.internalNeeds = needs;
    p.timeOfDay = world.getTimeOfDay();
    
    // Weather
    switch (world.getWeather()) {
        case Weather::Rain:
            p.weather = "rain";
            break;
        case Weather::Storm:
            p.weather = "storm";
            break;
        default:
            p.weather = "clear";
            break;
    }
    
    // Nearby tiles
    int centerX = static_cast<int>(position.x);
    int centerY = static_cast<int>(position.y);
    for (int dy = -5; dy <= 5; dy++) {
        for (int dx = -5; dx <= 5; dx++) {
            int x = centerX + dx;
            int y = centerY + dy;
            if (x >= 0 && x < world.getWidth() && y >= 0 && y < world.getHeight()) {
                const Tile& tile = world.getTile(x, y);
                std::string tileType = "grass"; // Simplified
                if (tile.type == TileType::BerryBush) tileType = "food";
                if (tile.type == TileType::Cave) tileType = "shelter";
                p.nearbyTiles.push_back({Vec2(static_cast<float>(x), static_cast<float>(y)), tileType});
            }
        }
    }
    
    // Nearby NPCs
    for (const auto& npc : allNPCs) {
        if (npc.getId() != id) {
            float dist = position.distance(npc.getPosition());
            if (dist < 20.0f) {
                p.nearbyNPCs.push_back({npc.getId(), npc.getPosition()});
            }
        }
    }
    
    // Memory recalls (if we have a behavior tree brain)
    if (auto* btBrain = dynamic_cast<BehaviorTreeBrain*>(brain.get())) {
        auto memories = btBrain->getMemory().getAllMemories();
        for (const auto& mem : memories) {
            if (mem.significance > 0.5f) {
                p.memoryRecalls.push_back(mem.type);
            }
        }
    }
    
    return p;
}

bool NPC::isNeuralBrain() const {
    // Forward declaration handled by include
    // Will check brain type at runtime
    return brain && (dynamic_cast<const class NeuralBrain*>(brain.get()) != nullptr);
}

} // namespace pw
