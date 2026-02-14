#include "DataLogger.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

namespace pw {

DataLogger::DataLogger(const std::string& outputDir) : outputDir(outputDir) {
    // Create output directory
    #ifdef _WIN32
        _mkdir(outputDir.c_str());
    #else
        mkdir(outputDir.c_str(), 0755);
    #endif
    
    // Open log files
    decisionsFile.open(outputDir + "/decisions.jsonl");
    eventsFile.open(outputDir + "/events.jsonl");
    
    if (!decisionsFile.is_open() || !eventsFile.is_open()) {
        std::cerr << "Warning: Could not open data log files" << std::endl;
    }
    
    // Write schema version
    json schemaInfo = {
        {"schema_version", SCHEMA_VERSION},
        {"timestamp", std::time(nullptr)}
    };
    if (decisionsFile.is_open()) {
        decisionsFile << schemaInfo.dump() << std::endl;
    }
    if (eventsFile.is_open()) {
        eventsFile << schemaInfo.dump() << std::endl;
    }
}

DataLogger::~DataLogger() {
    flush();
    decisionsFile.close();
    eventsFile.close();
}

void DataLogger::logDecision(Tick tick, EntityId npcId, const Perception& perception,
                              const Action& decision, const Outcome& outcome) {
    if (!decisionsFile.is_open()) return;
    
    json entry = {
        {"tick", tick},
        {"npc_id", std::string("npc_") + std::to_string(npcId)},
        {"perception", perceptionToJson(perception)},
        {"decision", actionToJson(decision)},
        {"outcome", outcomeToJson(outcome)}
    };
    
    decisionsFile << entry.dump() << std::endl;
    logCount++;
    
    // Periodic flush
    if (logCount % 100 == 0) {
        flush();
    }
}

void DataLogger::logEvent(Tick tick, const std::string& eventType, const json& eventData) {
    if (!eventsFile.is_open()) return;
    
    json entry = {
        {"tick", tick},
        {"event_type", eventType},
        {"data", eventData}
    };
    
    eventsFile << entry.dump() << std::endl;
}

void DataLogger::flush() {
    if (decisionsFile.is_open()) {
        decisionsFile.flush();
    }
    if (eventsFile.is_open()) {
        eventsFile.flush();
    }
}

json DataLogger::perceptionToJson(const Perception& p) const {
    json tiles = json::array();
    for (const auto& [pos, type] : p.nearbyTiles) {
        tiles.push_back({
            {"position", {pos.x, pos.y}},
            {"type", type}
        });
    }
    
    json npcs = json::array();
    for (const auto& [id, pos] : p.nearbyNPCs) {
        npcs.push_back({
            {"id", id},
            {"position", {pos.x, pos.y}}
        });
    }
    
    return {
        {"position", {p.position.x, p.position.y}},
        {"nearby_tiles", tiles},
        {"nearby_npcs", npcs},
        {"internal_needs", needsToJson(p.internalNeeds)},
        {"memory_recalls", p.memoryRecalls},
        {"weather", p.weather},
        {"time_of_day", p.timeOfDay}
    };
}

json DataLogger::actionToJson(const Action& a) const {
    return {
        {"type", a.toString()},
        {"target_position", {a.targetPosition.x, a.targetPosition.y}},
        {"target_entity", a.targetEntity}
    };
}

json DataLogger::outcomeToJson(const Outcome& o) const {
    return {
        {"needs_delta", o.needsDeltas},
        {"event", o.event}
    };
}

json DataLogger::needsToJson(const Needs& n) const {
    return {
        {"hunger", n.hunger},
        {"energy", n.energy},
        {"social", n.social},
        {"curiosity", n.curiosity},
        {"safety", n.safety}
    };
}

} // namespace pw
