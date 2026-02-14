#pragma once

#include "engine/Types.h"
#include "engine/Math.h"
#include <string>
#include <vector>
#include <memory>

namespace pw {

class World;

// NPC Needs system
struct Needs {
    float hunger = 0.5f;     // 0.0 = full, 1.0 = starving
    float energy = 0.5f;     // 0.0 = exhausted, 1.0 = need rest
    float social = 0.5f;     // 0.0 = lonely, 1.0 = need alone time
    float curiosity = 0.5f;  // 0.0 = content, 1.0 = need exploration
    float safety = 0.9f;     // 0.0 = safe, 1.0 = threatened
    
    void update(float dt);
    float getMostUrgent() const;
    std::string getMostUrgentName() const;
};

// Perception data for logging
struct Perception {
    Vec2 position;
    std::vector<std::pair<Vec2, std::string>> nearbyTiles;
    std::vector<std::pair<EntityId, Vec2>> nearbyNPCs;
    Needs internalNeeds;
    std::vector<std::string> memoryRecalls;
    std::string weather;
    float timeOfDay = 0.0f;
};

// Action types
enum class ActionType {
    Idle,
    Move,
    Forage,
    Eat,
    Rest,
    Explore,
    Socialize,
    BuildShelter,
    SeekShelter
};

struct Action {
    ActionType type = ActionType::Idle;
    Vec2 targetPosition;
    EntityId targetEntity = 0;
    
    std::string toString() const;
};

// Outcome after action
struct Outcome {
    std::map<std::string, float> needsDeltas;
    std::string event;
};

// Brain interface - allows swapping AI implementations
class IBrain {
public:
    virtual ~IBrain() = default;
    virtual Action decide(const Perception& perception, World& world) = 0;
    virtual void onOutcome(const Outcome& outcome) = 0;
};

} // namespace pw
