#include "ai/neural/NeuralBrain.h"
#include "world/World.h"
#include "world/Tile.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

namespace pw {

// EmotionalState implementation
void EmotionalState::clamp() {
    valence = std::max(-1.0f, std::min(1.0f, valence));
    arousal = std::max(-1.0f, std::min(1.0f, arousal));
    dominance = std::max(-1.0f, std::min(1.0f, dominance));
}

float EmotionalState::distance(const EmotionalState& other) const {
    float dv = valence - other.valence;
    float da = arousal - other.arousal;
    float dd = dominance - other.dominance;
    return std::sqrt(dv*dv + da*da + dd*dd);
}

// NeuralBrain implementation
NeuralBrain::NeuralBrain(EntityId ownerId, const std::string& modelPath)
    : ownerId(ownerId)
    , socialIntelligence(ownerId)
#ifdef HAS_ONNX_RUNTIME
    , memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
    lastActionProbs.resize(9, 0.0f);  // 9 action types
    
#ifdef HAS_ONNX_RUNTIME
    modelLoaded = loadModel(modelPath);
    if (!modelLoaded) {
        std::cerr << "Warning: NeuralBrain " << ownerId << " failed to load model: " 
                  << modelPath << std::endl;
    }
#else
    std::cerr << "Warning: ONNX Runtime not available, NeuralBrain will use fallback behavior" 
              << std::endl;
#endif
}

NeuralBrain::~NeuralBrain() = default;

bool NeuralBrain::loadModel(const std::string& modelPath) {
#ifdef HAS_ONNX_RUNTIME
    try {
        ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeuralBrain");
        ortSessionOptions = std::make_unique<Ort::SessionOptions>();
        ortSessionOptions->SetIntraOpNumThreads(1);
        ortSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        
        // Try to load the model
        ortSession = std::make_unique<Ort::Session>(*ortEnv, modelPath.c_str(), *ortSessionOptions);
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
#else
    (void)modelPath;
    return false;
#endif
}

Action NeuralBrain::decide(const Perception& perception, World& world) {
    (void)world;  // May be used for advanced queries
    
    // Update memory buffer with current perception
    updateMemoryBuffer(perception, 0);  // TODO: pass actual tick
    
    // Convert perception to vector
    std::vector<float> perceptionVec = perceptionToVector(perception);
    std::vector<float> memoryContext = getMemoryContext();
    
    // Run inference or fallback
    std::vector<float> actionProbs;
    std::vector<float> emotionOutput(3, 0.0f);
    
#ifdef HAS_ONNX_RUNTIME
    if (modelLoaded) {
        auto output = runInference(perceptionVec, memoryContext);
        if (output.size() >= 12) {  // 9 actions + 3 emotions
            actionProbs.assign(output.begin(), output.begin() + 9);
            emotionOutput.assign(output.begin() + 9, output.begin() + 12);
            
            // Update emotional state
            emotionalState.valence = emotionOutput[0];
            emotionalState.arousal = emotionOutput[1];
            emotionalState.dominance = emotionOutput[2];
            emotionalState.clamp();
        } else {
            // Fallback to uniform distribution
            actionProbs.assign(9, 1.0f / 9.0f);
        }
    } else
#endif
    {
        // Fallback: simple heuristic based on needs
        actionProbs.assign(9, 0.05f);
        
        const Needs& needs = perception.internalNeeds;
        if (needs.hunger > 0.7f) actionProbs[static_cast<int>(ActionType::Forage)] = 0.5f;
        else if (needs.energy > 0.7f) actionProbs[static_cast<int>(ActionType::Rest)] = 0.5f;
        else if (needs.social > 0.7f) actionProbs[static_cast<int>(ActionType::Socialize)] = 0.4f;
        else if (needs.curiosity > 0.6f) actionProbs[static_cast<int>(ActionType::Explore)] = 0.3f;
        else actionProbs[static_cast<int>(ActionType::Idle)] = 0.3f;
        
        // Normalize
        float sum = 0.0f;
        for (float p : actionProbs) sum += p;
        for (float& p : actionProbs) p /= sum;
    }
    
    // Modulate action probabilities by emotional state
    if (emotionalState.arousal > 0.5f) {
        // High arousal increases active actions
        actionProbs[static_cast<int>(ActionType::Explore)] *= 1.5f;
        actionProbs[static_cast<int>(ActionType::Move)] *= 1.3f;
    }
    if (emotionalState.valence < -0.5f) {
        // Negative emotion increases shelter-seeking
        actionProbs[static_cast<int>(ActionType::SeekShelter)] *= 2.0f;
        actionProbs[static_cast<int>(ActionType::Rest)] *= 1.5f;
    }
    
    // Normalize after modulation
    float sum = 0.0f;
    for (float p : actionProbs) sum += p;
    for (float& p : actionProbs) p /= sum;
    
    lastActionProbs = actionProbs;
    
    // Select action from distribution
    Action selectedAction = actionFromProbabilities(actionProbs, perception);
    
    // Cache for experience replay
    lastPerceptionVec = perceptionVec;
    lastMemoryContext = memoryContext;
    lastActionIndex = static_cast<int>(selectedAction.type);
    
    return selectedAction;
}

void NeuralBrain::onOutcome(const Outcome& outcome) {
    // Compute reward signal
    float reward = computeReward(outcome);
    
    // Store for online learning
    if (replayBuffer.size() < MAX_REPLAY_BUFFER && lastActionIndex >= 0 
        && !lastPerceptionVec.empty()) {
        ExperienceReplay exp;
        exp.perceptionVec = lastPerceptionVec;
        exp.actionIndex = lastActionIndex;
        exp.reward = reward;
        exp.memoryContext = lastMemoryContext;
        replayBuffer.push_back(std::move(exp));
        
        // Apply online update when buffer has enough samples
        if (replayBuffer.size() >= 10) {
            applyOnlineUpdate(replayBuffer.back());
        }
    }
    
    // Update emotional state based on outcome
    for (const auto& [need, delta] : outcome.needsDeltas) {
        if (delta < 0) {  // Need satisfied
            emotionalState.valence += 0.1f;
            emotionalState.arousal -= 0.05f;
        } else {  // Need increased
            emotionalState.valence -= 0.05f;
            emotionalState.arousal += 0.1f;
        }
    }
    
    if (!outcome.event.empty()) {
        if (outcome.event.find("danger") != std::string::npos ||
            outcome.event.find("attacked") != std::string::npos) {
            emotionalState.valence -= 0.3f;
            emotionalState.arousal += 0.4f;
            emotionalState.dominance -= 0.2f;
        } else if (outcome.event.find("food") != std::string::npos ||
                   outcome.event.find("social") != std::string::npos) {
            emotionalState.valence += 0.2f;
        }
    }
    
    emotionalState.clamp();
}

std::vector<float> NeuralBrain::perceptionToVector(const Perception& perception) const {
    std::vector<float> vec;
    vec.reserve(20);  // Approximate size
    
    // Position (2)
    vec.push_back(perception.position.x / static_cast<float>(WORLD_WIDTH));
    vec.push_back(perception.position.y / static_cast<float>(WORLD_HEIGHT));
    
    // Needs (5)
    vec.push_back(perception.internalNeeds.hunger);
    vec.push_back(perception.internalNeeds.energy);
    vec.push_back(perception.internalNeeds.social);
    vec.push_back(perception.internalNeeds.curiosity);
    vec.push_back(perception.internalNeeds.safety);
    
    // Time and weather (2)
    vec.push_back(perception.timeOfDay);
    vec.push_back(perception.weather == "rain" ? 1.0f : 0.0f);
    
    // Nearby tiles (one-hot encoded, simplified to counts)
    int waterCount = 0, foodCount = 0, shelterCount = 0;
    for (const auto& [pos, type] : perception.nearbyTiles) {
        if (type == "Water") waterCount++;
        else if (type == "BerryBush" || type == "Tree") foodCount++;
        else if (type == "Cave" || type == "Shelter") shelterCount++;
    }
    vec.push_back(std::min(1.0f, waterCount / 5.0f));
    vec.push_back(std::min(1.0f, foodCount / 5.0f));
    vec.push_back(std::min(1.0f, shelterCount / 3.0f));
    
    // Nearby NPCs count (1)
    vec.push_back(std::min(1.0f, perception.nearbyNPCs.size() / 5.0f));
    
    // Emotional state (3)
    vec.push_back(emotionalState.valence);
    vec.push_back(emotionalState.arousal);
    vec.push_back(emotionalState.dominance);
    
    // Pad to fixed size if needed
    while (vec.size() < 20) {
        vec.push_back(0.0f);
    }
    
    return vec;
}

std::vector<float> NeuralBrain::getMemoryContext() const {
    std::vector<float> context;
    context.reserve(MAX_MEMORY_BUFFER * MEMORY_EMBEDDING_DIM);
    
    // Simplified: encode recent memories
    for (size_t i = 0; i < std::min(memoryBuffer.size(), MAX_MEMORY_BUFFER); ++i) {
        const auto& mem = memoryBuffer[i];
        
        // Simple embedding: position + significance + type encoding
        std::vector<float> embedding(MEMORY_EMBEDDING_DIM, 0.0f);
        embedding[0] = mem.memory.location.x / static_cast<float>(WORLD_WIDTH);
        embedding[1] = mem.memory.location.y / static_cast<float>(WORLD_HEIGHT);
        embedding[2] = mem.memory.significance;
        embedding[3] = mem.attentionWeight;
        
        // Type encoding (one-hot-ish)
        if (mem.memory.type == "food") embedding[4] = 1.0f;
        else if (mem.memory.type == "danger") embedding[5] = 1.0f;
        else if (mem.memory.type == "npc") embedding[6] = 1.0f;
        else if (mem.memory.type == "shelter") embedding[7] = 1.0f;
        
        context.insert(context.end(), embedding.begin(), embedding.end());
    }
    
    // Pad if needed
    while (context.size() < MAX_MEMORY_BUFFER * MEMORY_EMBEDDING_DIM) {
        context.push_back(0.0f);
    }
    
    return context;
}

Action NeuralBrain::actionFromProbabilities(const std::vector<float>& probs, 
                                            const Perception& perception) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    
    int actionIdx = dist(gen);
    ActionType actionType = static_cast<ActionType>(actionIdx);
    
    Action action;
    action.type = actionType;
    
    // Set target based on action type
    switch (actionType) {
        case ActionType::Move:
        case ActionType::Explore:
            // Random nearby position
            action.targetPosition = perception.position + Vec2{
                static_cast<float>((gen() % 40) - 20),
                static_cast<float>((gen() % 40) - 20)
            };
            break;
        case ActionType::Forage:
        case ActionType::Eat:
            // Move towards food if seen
            for (const auto& [pos, type] : perception.nearbyTiles) {
                if (type == "BerryBush" || type == "Tree") {
                    action.targetPosition = pos;
                    break;
                }
            }
            break;
        case ActionType::Socialize:
            // Move towards nearest NPC
            if (!perception.nearbyNPCs.empty()) {
                action.targetEntity = perception.nearbyNPCs[0].first;
                action.targetPosition = perception.nearbyNPCs[0].second;
            }
            break;
        case ActionType::SeekShelter:
            // Move towards shelter
            for (const auto& [pos, type] : perception.nearbyTiles) {
                if (type == "Cave" || type == "Shelter") {
                    action.targetPosition = pos;
                    break;
                }
            }
            break;
        default:
            action.targetPosition = perception.position;
            break;
    }
    
    return action;
}

void NeuralBrain::updateMemoryBuffer(const Perception& perception, Tick currentTick) {
    // Add significant perceptions to memory buffer
    
    // Add food sightings
    for (const auto& [pos, type] : perception.nearbyTiles) {
        if (type == "BerryBush" || type == "Tree") {
            float significance = perception.internalNeeds.hunger * 1.5f;
            MemoryEntry mem("food", pos, currentTick, significance);
            
            // Create embedding (simplified)
            std::vector<float> embedding(MEMORY_EMBEDDING_DIM, 0.0f);
            embedding[0] = pos.x / static_cast<float>(WORLD_WIDTH);
            embedding[1] = pos.y / static_cast<float>(WORLD_HEIGHT);
            embedding[2] = significance;
            
            memoryBuffer.emplace_back(mem, embedding);
        }
    }
    
    // Add NPC encounters
    for (const auto& [npcId, pos] : perception.nearbyNPCs) {
        float significance = perception.internalNeeds.social * 1.2f;
        MemoryEntry mem("npc", pos, currentTick, significance);
        
        std::vector<float> embedding(MEMORY_EMBEDDING_DIM, 0.0f);
        embedding[0] = pos.x / static_cast<float>(WORLD_WIDTH);
        embedding[1] = pos.y / static_cast<float>(WORLD_HEIGHT);
        embedding[2] = significance;
        embedding[6] = 1.0f;  // NPC type marker
        
        memoryBuffer.emplace_back(mem, embedding);
    }
    
    // Maintain buffer size
    if (memoryBuffer.size() > MAX_MEMORY_BUFFER) {
        // Remove lowest significance memories
        std::sort(memoryBuffer.begin(), memoryBuffer.end(),
                  [](const EpisodicMemory& a, const EpisodicMemory& b) {
                      return a.memory.significance < b.memory.significance;
                  });
        memoryBuffer.erase(memoryBuffer.begin(), 
                          memoryBuffer.begin() + (memoryBuffer.size() - MAX_MEMORY_BUFFER));
    }
}

void NeuralBrain::computeMemoryAttention(const std::vector<float>& queryVec) {
    // Simplified attention: dot product similarity
    for (auto& mem : memoryBuffer) {
        float similarity = 0.0f;
        size_t minSize = std::min(queryVec.size(), mem.embedding.size());
        for (size_t i = 0; i < minSize; ++i) {
            similarity += queryVec[i] * mem.embedding[i];
        }
        mem.attentionWeight = std::max(0.0f, similarity);
    }
    
    // Softmax normalization
    float maxWeight = -1e9f;
    for (const auto& mem : memoryBuffer) {
        maxWeight = std::max(maxWeight, mem.attentionWeight);
    }
    
    float sumExp = 0.0f;
    for (auto& mem : memoryBuffer) {
        mem.attentionWeight = std::exp(mem.attentionWeight - maxWeight);
        sumExp += mem.attentionWeight;
    }
    
    if (sumExp > 0.0f) {
        for (auto& mem : memoryBuffer) {
            mem.attentionWeight /= sumExp;
        }
    }
}

void NeuralBrain::decayMemories(Tick currentTick) {
    for (auto& mem : memoryBuffer) {
        Tick age = currentTick - mem.memory.timestamp;
        float decayFactor = 1.0f - (age / 10000.0f);  // Decay over 10000 ticks
        mem.memory.significance *= std::max(0.1f, decayFactor);
    }
}

void NeuralBrain::triggerProustianRecall(const std::vector<float>& currentPerception) {
    // Check if current perception triggers a strong match with old memory
    computeMemoryAttention(currentPerception);
    
    for (auto& mem : memoryBuffer) {
        // If an old, decayed memory gets high attention, boost its significance
        if (mem.attentionWeight > 0.3f && mem.memory.significance < 0.3f) {
            // Flashback! This memory resurfaces
            mem.memory.significance = std::min(1.0f, mem.memory.significance + 0.5f);
            
            // Emotional impact
            emotionalState.arousal += 0.2f;
            emotionalState.clamp();
        }
    }
}

std::vector<float> NeuralBrain::runInference(const std::vector<float>& perceptionVec,
                                             const std::vector<float>& memoryContext) {
#ifdef HAS_ONNX_RUNTIME
    if (!modelLoaded || !ortSession) {
        return std::vector<float>(12, 0.0f);  // Return zeros
    }
    
    try {
        // Prepare input tensors with correct shapes
        // Perception: (batch=1, perception_dim=20)
        std::vector<int64_t> perceptionShape = {1, static_cast<int64_t>(perceptionVec.size())};
        
        // Memory: (batch=1, seq_len=50, embedding_dim=32)
        // Memory context is flattened (50 * 32 = 1600), need to reshape
        std::vector<int64_t> memoryShape = {
            1,  // batch
            static_cast<int64_t>(MAX_MEMORY_BUFFER),  // sequence length = 50
            static_cast<int64_t>(MEMORY_EMBEDDING_DIM)  // embedding dim = 32
        };
        
        Ort::Value perceptionTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(perceptionVec.data()), perceptionVec.size(),
            perceptionShape.data(), perceptionShape.size());
            
        Ort::Value memoryTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(memoryContext.data()), memoryContext.size(),
            memoryShape.data(), memoryShape.size());
        
        // Input names
        const char* inputNames[] = {"perception", "memory"};
        const char* outputNames[] = {"output"};
        Ort::Value inputTensors[] = {std::move(perceptionTensor), std::move(memoryTensor)};
        
        // Run inference
        auto outputTensors = ortSession->Run(
            Ort::RunOptions{nullptr}, inputNames, inputTensors, 2, outputNames, 1);
        
        // Extract output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t outputSize = 1;
        for (auto dim : outputShape) outputSize *= dim;
        
        return std::vector<float>(outputData, outputData + outputSize);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return std::vector<float>(12, 0.0f);
    }
#else
    (void)perceptionVec;
    (void)memoryContext;
    return std::vector<float>(12, 0.0f);
#endif
}

float NeuralBrain::computeReward(const Outcome& outcome) const {
    float reward = 0.0f;
    
    for (const auto& [need, delta] : outcome.needsDeltas) {
        if (delta < 0) {  // Need satisfied
            reward += -delta;  // Positive reward
        } else {  // Need increased
            reward += -delta * 0.5f;  // Negative reward
        }
    }
    
    if (!outcome.event.empty()) {
        if (outcome.event.find("danger") != std::string::npos) {
            reward -= 2.0f;
        } else if (outcome.event.find("food") != std::string::npos) {
            reward += 1.0f;
        } else if (outcome.event.find("social") != std::string::npos) {
            reward += 0.5f;
        }
    }
    
    return reward;
}

void NeuralBrain::updateFromExperience(const Perception& perception, const Action& action,
                                       const Outcome& outcome, float reward) {
    // Store in replay buffer
    ExperienceReplay exp;
    exp.perceptionVec = perceptionToVector(perception);
    exp.actionIndex = static_cast<int>(action.type);
    exp.reward = reward;
    exp.memoryContext = getMemoryContext();
    
    if (replayBuffer.size() >= MAX_REPLAY_BUFFER) {
        // Remove oldest experience
        replayBuffer.erase(replayBuffer.begin());
    }
    replayBuffer.push_back(std::move(exp));
    
    // Update emotional state based on reward
    emotionalState.valence += reward * 0.05f;
    emotionalState.clamp();
}

void NeuralBrain::applyOnlineUpdate(const ExperienceReplay& experience) {
    // Reward-modulated update: adjust emotional state based on experience quality
    static constexpr float VALENCE_SCALE = 10.0f;
    static constexpr float AROUSAL_POSITIVE_SCALE = 5.0f;
    static constexpr float AROUSAL_NEGATIVE_SCALE = 3.0f;
    static constexpr float DOMINANCE_SCALE = 2.0f;
    static constexpr float REWARD_THRESHOLD = 0.5f;
    
    float rewardSignal = experience.reward;
    
    // Modulate emotional state as a form of "soft learning"
    emotionalState.valence += rewardSignal * learningRate * VALENCE_SCALE;
    
    // High-reward actions in active states increase arousal tendency
    if (rewardSignal > REWARD_THRESHOLD) {
        emotionalState.arousal += learningRate * AROUSAL_POSITIVE_SCALE;
        emotionalState.dominance += learningRate * DOMINANCE_SCALE;
    } else if (rewardSignal < -REWARD_THRESHOLD) {
        emotionalState.arousal -= learningRate * AROUSAL_NEGATIVE_SCALE;
        emotionalState.dominance -= learningRate * DOMINANCE_SCALE;
    }
    
    emotionalState.clamp();
}

void NeuralBrain::saveState(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) return;
    
    nlohmann::json state;
    
    // Save emotional state
    state["emotional_state"] = {
        {"valence", emotionalState.valence},
        {"arousal", emotionalState.arousal},
        {"dominance", emotionalState.dominance}
    };
    
    // Save memory buffer
    nlohmann::json memoriesJson = nlohmann::json::array();
    for (const auto& mem : memoryBuffer) {
        nlohmann::json memJson;
        memJson["type"] = mem.memory.type;
        memJson["location"] = {{"x", mem.memory.location.x}, {"y", mem.memory.location.y}};
        memJson["timestamp"] = mem.memory.timestamp;
        memJson["significance"] = mem.memory.significance;
        memJson["attention_weight"] = mem.attentionWeight;
        memJson["embedding"] = mem.embedding;
        memoriesJson.push_back(memJson);
    }
    state["memory_buffer"] = memoriesJson;
    
    // Save social relationships
    nlohmann::json socialJson = nlohmann::json::object();
    for (const auto& [id, rel] : socialIntelligence.getAllRelationships()) {
        nlohmann::json relJson;
        relJson["npc_id"] = rel.npcId;
        relJson["trust"] = rel.trust;
        relJson["affinity"] = rel.affinity;
        relJson["last_interaction"] = rel.lastInteraction;
        relJson["embedding"] = rel.embedding;
        socialJson[std::to_string(id)] = relJson;
    }
    state["social_relationships"] = socialJson;
    
    // Save experience replay buffer size (not full buffer, to keep save files small)
    state["replay_buffer_size"] = replayBuffer.size();
    state["owner_id"] = ownerId;
    
    file << state.dump(2);
    file.close();
}

void NeuralBrain::loadState(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return;
    
    nlohmann::json state;
    try {
        file >> state;
    } catch (const nlohmann::json::parse_error&) {
        return;
    }
    file.close();
    
    // Load emotional state
    if (state.contains("emotional_state")) {
        const auto& es = state["emotional_state"];
        emotionalState.valence = es.value("valence", 0.0f);
        emotionalState.arousal = es.value("arousal", 0.0f);
        emotionalState.dominance = es.value("dominance", 0.0f);
        emotionalState.clamp();
    }
    
    // Load memory buffer
    if (state.contains("memory_buffer")) {
        memoryBuffer.clear();
        for (const auto& memJson : state["memory_buffer"]) {
            MemoryEntry mem;
            mem.type = memJson.value("type", "");
            if (memJson.contains("location")) {
                mem.location.x = memJson["location"].value("x", 0.0f);
                mem.location.y = memJson["location"].value("y", 0.0f);
            }
            mem.timestamp = memJson.value("timestamp", static_cast<Tick>(0));
            mem.significance = memJson.value("significance", 0.0f);
            
            std::vector<float> embedding(MEMORY_EMBEDDING_DIM, 0.0f);
            if (memJson.contains("embedding")) {
                const auto& embJson = memJson["embedding"];
                for (size_t i = 0; i < std::min(embJson.size(), static_cast<size_t>(MEMORY_EMBEDDING_DIM)); ++i) {
                    embedding[i] = embJson[i].get<float>();
                }
            }
            
            EpisodicMemory epMem(mem, embedding);
            epMem.attentionWeight = memJson.value("attention_weight", 0.0f);
            memoryBuffer.push_back(std::move(epMem));
        }
    }
    
    // Load social relationships
    if (state.contains("social_relationships")) {
        for (const auto& [idStr, relJson] : state["social_relationships"].items()) {
            try {
                EntityId npcId = static_cast<EntityId>(std::stoul(idStr));
                socialIntelligence.recordInteraction(npcId, "neutral", 0.0f, 
                                                      relJson.value("last_interaction", static_cast<Tick>(0)));
            } catch (const std::exception&) {
                // Skip malformed entries
            }
        }
    }
}

} // namespace pw
