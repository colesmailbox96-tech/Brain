#pragma once

#include "ai/interface/IBrain.h"
#include "entities/NPC.h"
#include "engine/Types.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace pw {

using json = nlohmann::json;

class DataLogger {
public:
    DataLogger(const std::string& outputDir = "data_logs");
    ~DataLogger();
    
    void logDecision(Tick tick, EntityId npcId, const Perception& perception,
                     const Action& decision, const Outcome& outcome);
    
    void logEvent(Tick tick, const std::string& eventType, const json& eventData);
    
    void flush();
    
    static constexpr const char* SCHEMA_VERSION = "1.0.0";

private:
    std::string outputDir;
    std::ofstream decisionsFile;
    std::ofstream eventsFile;
    int logCount = 0;
    
    json perceptionToJson(const Perception& p) const;
    json actionToJson(const Action& a) const;
    json outcomeToJson(const Outcome& o) const;
    json needsToJson(const Needs& n) const;
};

} // namespace pw
