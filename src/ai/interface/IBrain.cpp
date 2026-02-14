#include "IBrain.h"
#include <algorithm>

namespace pw {

void Needs::update(float dt) {
    // Needs naturally increase over time
    hunger = std::min(1.0f, hunger + dt * 0.05f);
    energy = std::min(1.0f, energy + dt * 0.03f);
    social = std::min(1.0f, social + dt * 0.02f);
    curiosity = std::min(1.0f, curiosity + dt * 0.01f);
    
    // Safety naturally decreases (becomes safer)
    safety = std::max(0.0f, safety - dt * 0.1f);
}

float Needs::getMostUrgent() const {
    return std::max({hunger, energy, social, curiosity, 1.0f - safety});
}

std::string Needs::getMostUrgentName() const {
    float maxNeed = 0.0f;
    std::string name = "none";
    
    if (hunger > maxNeed) {
        maxNeed = hunger;
        name = "hunger";
    }
    if (energy > maxNeed) {
        maxNeed = energy;
        name = "energy";
    }
    if (social > maxNeed) {
        maxNeed = social;
        name = "social";
    }
    if (curiosity > maxNeed) {
        maxNeed = curiosity;
        name = "curiosity";
    }
    if ((1.0f - safety) > maxNeed) {
        name = "safety";
    }
    
    return name;
}

std::string Action::toString() const {
    switch (type) {
        case ActionType::Idle: return "idle";
        case ActionType::Move: return "move";
        case ActionType::Forage: return "forage";
        case ActionType::Eat: return "eat";
        case ActionType::Rest: return "rest";
        case ActionType::Explore: return "explore";
        case ActionType::Socialize: return "socialize";
        case ActionType::BuildShelter: return "build_shelter";
        case ActionType::SeekShelter: return "seek_shelter";
        default: return "unknown";
    }
}

} // namespace pw
