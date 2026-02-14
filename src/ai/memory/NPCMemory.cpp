#include "NPCMemory.h"
#include <algorithm>

namespace pw {

void NPCMemory::addMemory(const std::string& type, Vec2 location, Tick currentTick, float significance) {
    memories.push_back(MemoryEntry(type, location, currentTick, significance));
    
    // Keep only most significant memories if we exceed max
    if (memories.size() > MAX_MEMORIES) {
        std::sort(memories.begin(), memories.end(),
            [](const MemoryEntry& a, const MemoryEntry& b) {
                return a.significance > b.significance;
            });
        memories.resize(MAX_MEMORIES);
    }
}

std::vector<MemoryEntry> NPCMemory::recall(const std::string& type, int maxResults) const {
    std::vector<MemoryEntry> result;
    
    for (const auto& mem : memories) {
        if (mem.type == type) {
            result.push_back(mem);
        }
    }
    
    // Sort by significance
    std::sort(result.begin(), result.end(),
        [](const MemoryEntry& a, const MemoryEntry& b) {
            return a.significance > b.significance;
        });
    
    if (result.size() > static_cast<size_t>(maxResults)) {
        result.resize(maxResults);
    }
    
    return result;
}

std::vector<MemoryEntry> NPCMemory::recallNearby(Vec2 position, float radius, int maxResults) const {
    std::vector<MemoryEntry> result;
    
    for (const auto& mem : memories) {
        if (mem.location.distance(position) <= radius) {
            result.push_back(mem);
        }
    }
    
    // Sort by distance
    std::sort(result.begin(), result.end(),
        [position](const MemoryEntry& a, const MemoryEntry& b) {
            return a.location.distance(position) < b.location.distance(position);
        });
    
    if (result.size() > static_cast<size_t>(maxResults)) {
        result.resize(maxResults);
    }
    
    return result;
}

void NPCMemory::decay(Tick currentTick) {
    // Decay significance over time
    for (auto& mem : memories) {
        Tick age = currentTick - mem.timestamp;
        float decayRate = 0.001f;
        mem.significance *= (1.0f - decayRate * age);
        
        if (mem.significance < 0.01f) {
            mem.significance = 0.01f;
        }
    }
}

} // namespace pw
