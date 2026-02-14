#include "BehaviorTreeBrain.h"
#include "world/World.h"
#include <cmath>

namespace pw {

BehaviorTreeBrain::BehaviorTreeBrain(EntityId ownerId)
    : ownerId(ownerId), rng(std::random_device{}()) {
}

Action BehaviorTreeBrain::decide(const Perception& perception, World& world) {
    return decideBasedOnNeeds(perception, world);
}

void BehaviorTreeBrain::onOutcome(const Outcome& outcome) {
    // Learn from outcome (future: used for learning systems)
}

Action BehaviorTreeBrain::decideBasedOnNeeds(const Perception& perception, World& world) {
    const Needs& needs = perception.internalNeeds;
    
    // Check weather - seek shelter if raining
    if (perception.weather == "rain" || perception.weather == "storm") {
        if (needs.safety < 0.7f) { // Not urgent safety need
            return seekShelter(perception, world);
        }
    }
    
    // Prioritize based on most urgent need
    std::string urgentNeed = needs.getMostUrgentName();
    
    if (needs.hunger > 0.7f) {
        return forageForFood(perception, world);
    } else if (needs.energy > 0.7f) {
        return seekRest(perception, world);
    } else if (needs.safety > 0.7f) {
        return seekShelter(perception, world);
    } else if (needs.social > 0.6f) {
        return socialize(perception, world);
    } else if (needs.curiosity > 0.6f) {
        return explore(perception, world);
    }
    
    // Default: explore
    return explore(perception, world);
}

Action BehaviorTreeBrain::forageForFood(const Perception& perception, World& world) {
    // Check memory for known food sources
    auto foodMemories = memory.recall("food", 3);
    
    Vec2 target;
    bool foundTarget = false;
    
    // Try remembered locations first
    for (const auto& mem : foodMemories) {
        if (mem.location.distance(perception.position) < 100.0f) {
            target = mem.location;
            foundTarget = true;
            break;
        }
    }
    
    // Search for nearby food
    if (!foundTarget) {
        target = findNearestTile(perception, world, TileType::BerryBush, 50.0f);
        if (target.x >= 0 && target.y >= 0) {
            foundTarget = true;
            memory.addMemory("food", target, 0, 1.0f);
        }
    }
    
    if (foundTarget) {
        float dist = perception.position.distance(target);
        if (dist < 1.5f) {
            // At food source, eat
            Action action;
            action.type = ActionType::Eat;
            action.targetPosition = target;
            return action;
        } else {
            // Move to food
            Action action;
            action.type = ActionType::Move;
            action.targetPosition = target;
            return action;
        }
    }
    
    // No food found, explore
    return explore(perception, world);
}

Action BehaviorTreeBrain::seekRest(const Perception& perception, World& world) {
    // Look for shelter or safe spot
    auto shelterMemories = memory.recall("shelter", 3);
    
    Vec2 target;
    bool foundTarget = false;
    
    for (const auto& mem : shelterMemories) {
        if (mem.location.distance(perception.position) < 100.0f) {
            target = mem.location;
            foundTarget = true;
            break;
        }
    }
    
    if (!foundTarget) {
        target = findNearestTile(perception, world, TileType::Cave, 50.0f);
        if (target.x >= 0 && target.y >= 0) {
            foundTarget = true;
            memory.addMemory("shelter", target, 0, 1.0f);
        }
    }
    
    if (foundTarget) {
        float dist = perception.position.distance(target);
        if (dist < 2.0f) {
            Action action;
            action.type = ActionType::Rest;
            return action;
        } else {
            Action action;
            action.type = ActionType::Move;
            action.targetPosition = target;
            return action;
        }
    }
    
    // Just rest where we are
    Action action;
    action.type = ActionType::Rest;
    return action;
}

Action BehaviorTreeBrain::socialize(const Perception& perception, World& world) {
    // Find nearest NPC
    if (!perception.nearbyNPCs.empty()) {
        float minDist = 1000.0f;
        Vec2 nearest;
        
        for (const auto& [id, pos] : perception.nearbyNPCs) {
            float dist = perception.position.distance(pos);
            if (dist < minDist) {
                minDist = dist;
                nearest = pos;
            }
        }
        
        if (minDist < 3.0f) {
            Action action;
            action.type = ActionType::Socialize;
            action.targetPosition = nearest;
            return action;
        } else {
            Action action;
            action.type = ActionType::Move;
            action.targetPosition = nearest;
            return action;
        }
    }
    
    return explore(perception, world);
}

Action BehaviorTreeBrain::explore(const Perception& perception, World& world) {
    Vec2 target = findRandomWalkableNearby(perception, world, 30.0f);
    
    Action action;
    action.type = ActionType::Explore;
    action.targetPosition = target;
    return action;
}

Action BehaviorTreeBrain::seekShelter(const Perception& perception, World& world) {
    Vec2 target = findNearestTile(perception, world, TileType::Cave, 50.0f);
    
    if (target.x < 0 || target.y < 0) {
        target = findNearestTile(perception, world, TileType::Tree, 30.0f);
    }
    
    if (target.x >= 0 && target.y >= 0) {
        Action action;
        action.type = ActionType::SeekShelter;
        action.targetPosition = target;
        return action;
    }
    
    Action action;
    action.type = ActionType::Idle;
    return action;
}

Vec2 BehaviorTreeBrain::findNearestTile(const Perception& perception, World& world, TileType type, float maxDist) {
    int centerX = static_cast<int>(perception.position.x);
    int centerY = static_cast<int>(perception.position.y);
    int searchRadius = static_cast<int>(maxDist);
    
    float minDist = maxDist * maxDist;
    Vec2 nearest(-1, -1);
    
    for (int dy = -searchRadius; dy <= searchRadius; dy++) {
        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            int x = centerX + dx;
            int y = centerY + dy;
            
            if (x < 0 || x >= world.getWidth() || y < 0 || y >= world.getHeight()) {
                continue;
            }
            
            const Tile& tile = world.getTile(x, y);
            if (tile.type == type) {
                float distSq = dx * dx + dy * dy;
                if (distSq < minDist) {
                    minDist = distSq;
                    nearest = Vec2(static_cast<float>(x), static_cast<float>(y));
                }
            }
        }
    }
    
    return nearest;
}

Vec2 BehaviorTreeBrain::findRandomWalkableNearby(const Perception& perception, World& world, float radius) {
    std::uniform_real_distribution<float> angleDist(0.0f, 6.28318f);
    std::uniform_real_distribution<float> radiusDist(radius * 0.5f, radius);
    
    for (int attempt = 0; attempt < 10; attempt++) {
        float angle = angleDist(rng);
        float r = radiusDist(rng);
        
        Vec2 target(
            perception.position.x + std::cos(angle) * r,
            perception.position.y + std::sin(angle) * r
        );
        
        int x = static_cast<int>(target.x);
        int y = static_cast<int>(target.y);
        
        if (world.isWalkable(x, y)) {
            return target;
        }
    }
    
    return perception.position;
}

} // namespace pw
