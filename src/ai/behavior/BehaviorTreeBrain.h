#pragma once

#include "ai/interface/IBrain.h"
#include "ai/memory/NPCMemory.h"
#include "ai/behavior/Pathfinder.h"
#include "world/Tile.h"
#include <random>

namespace pw {

class BehaviorTreeBrain : public IBrain {
public:
    BehaviorTreeBrain(EntityId ownerId);
    
    Action decide(const Perception& perception, World& world) override;
    void onOutcome(const Outcome& outcome) override;
    
    NPCMemory& getMemory() { return memory; }

private:
    EntityId ownerId;
    NPCMemory memory;
    std::mt19937 rng;
    
    Action currentAction;
    std::vector<Vec2> currentPath;
    int pathIndex = 0;
    
    // Decision making
    Action decideBasedOnNeeds(const Perception& perception, World& world);
    Action forageForFood(const Perception& perception, World& world);
    Action seekRest(const Perception& perception, World& world);
    Action socialize(const Perception& perception, World& world);
    Action explore(const Perception& perception, World& world);
    Action seekShelter(const Perception& perception, World& world);
    
    // Utilities
    Vec2 findNearestTile(const Perception& perception, World& world, TileType type, float maxDist = 50.0f);
    Vec2 findRandomWalkableNearby(const Perception& perception, World& world, float radius = 20.0f);
};

} // namespace pw
