#pragma once

#include "engine/Types.h"
#include "engine/Math.h"
#include "ai/interface/IBrain.h"
#include <memory>
#include <string>

namespace pw {

class World;

enum class Mood {
    Happy,
    Neutral,
    Sad,
    Anxious,
    Excited
};

class NPC {
public:
    NPC(EntityId id, Vec2 position);
    
    void update(float dt, World& world, Tick currentTick);
    
    EntityId getId() const { return id; }
    Vec2 getPosition() const { return position; }
    Needs& getNeeds() { return needs; }
    const Needs& getNeeds() const { return needs; }
    Mood getMood() const { return mood; }
    Color getColor() const { return color; }
    const Action& getCurrentAction() const { return currentAction; }
    
    void setBrain(std::unique_ptr<IBrain> newBrain);
    IBrain* getBrain() { return brain.get(); }
    
    // For data logging
    Perception gatherPerception(const World& world, const std::vector<NPC>& allNPCs) const;

private:
    EntityId id;
    Vec2 position;
    Vec2 velocity;
    Needs needs;
    Mood mood = Mood::Neutral;
    Color color;
    float speed = 10.0f; // tiles per second
    
    std::unique_ptr<IBrain> brain;
    Action currentAction;
    Vec2 moveTarget;
    
    void updateNeeds(float dt);
    void updateMood();
    void executeAction(float dt, World& world);
    void moveTowards(Vec2 target, float dt);
};

} // namespace pw
