#pragma once

#include "engine/Math.h"
#include "engine/Types.h"
#include <vector>
#include <map>
#include <string>

namespace pw {

struct MemoryEntry {
    std::string type; // "food", "danger", "npc", "shelter"
    Vec2 location;
    Tick timestamp;
    float significance = 1.0f;
    
    MemoryEntry() = default;
    MemoryEntry(const std::string& t, Vec2 loc, Tick ts, float sig = 1.0f)
        : type(t), location(loc), timestamp(ts), significance(sig) {}
};

class NPCMemory {
public:
    void addMemory(const std::string& type, Vec2 location, Tick currentTick, float significance = 1.0f);
    std::vector<MemoryEntry> recall(const std::string& type, int maxResults = 5) const;
    std::vector<MemoryEntry> recallNearby(Vec2 position, float radius, int maxResults = 5) const;
    void decay(Tick currentTick);
    
    const std::vector<MemoryEntry>& getAllMemories() const { return memories; }

private:
    std::vector<MemoryEntry> memories;
    static constexpr int MAX_MEMORIES = 100;
};

} // namespace pw
