#pragma once

#include "ai/interface/IBrain.h"
#include "ai/memory/NPCMemory.h"
#include "ai/social/SocialIntelligence.h"
#include <vector>
#include <string>
#include <memory>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace pw {

// Emotional state representation
struct EmotionalState {
    float valence = 0.0f;      // -1.0 (negative) to +1.0 (positive)
    float arousal = 0.0f;      // -1.0 (calm) to +1.0 (excited)
    float dominance = 0.0f;    // -1.0 (submissive) to +1.0 (dominant)
    
    void clamp();
    float distance(const EmotionalState& other) const;
};

// Memory buffer entry with embedding
struct EpisodicMemory {
    MemoryEntry memory;
    std::vector<float> embedding;  // Neural representation
    float attentionWeight = 0.0f;  // Last attention weight from transformer
    
    EpisodicMemory() = default;
    EpisodicMemory(const MemoryEntry& mem, const std::vector<float>& emb)
        : memory(mem), embedding(emb) {}
};

class NeuralBrain : public IBrain {
public:
    NeuralBrain(EntityId ownerId, const std::string& modelPath = "models/npc_brain.onnx");
    ~NeuralBrain();
    
    Action decide(const Perception& perception, World& world) override;
    void onOutcome(const Outcome& outcome) override;
    
    // Access internal state for debugging
    const EmotionalState& getEmotionalState() const { return emotionalState; }
    const std::vector<EpisodicMemory>& getMemoryBuffer() const { return memoryBuffer; }
    const std::vector<float>& getLastActionProbs() const { return lastActionProbs; }
    
    // Memory management
    NPCMemory& getMemory() { return memory; }
    
    // Social intelligence
    SocialIntelligence& getSocialIntelligence() { return socialIntelligence; }
    const SocialIntelligence& getSocialIntelligence() const { return socialIntelligence; }
    
    // Online learning
    void updateFromExperience(const Perception& perception, const Action& action, 
                              const Outcome& outcome, float reward);
    
    // Serialization (JSON-based)
    void saveState(const std::string& filepath) const;
    void loadState(const std::string& filepath);

private:
#ifdef HAS_ONNX_RUNTIME
    std::unique_ptr<Ort::Env> ortEnv;
    std::unique_ptr<Ort::Session> ortSession;
    std::unique_ptr<Ort::SessionOptions> ortSessionOptions;
    Ort::MemoryInfo memoryInfo{nullptr};
#endif
    
    EntityId ownerId;
    NPCMemory memory;
    SocialIntelligence socialIntelligence;
    bool modelLoaded = false;
    
    // Neural state
    EmotionalState emotionalState;
    std::vector<EpisodicMemory> memoryBuffer;
    static constexpr size_t MAX_MEMORY_BUFFER = 50;
    static constexpr size_t MEMORY_EMBEDDING_DIM = 32;
    
    // Inference output cache
    std::vector<float> lastActionProbs;
    
    // Online learning
    struct ExperienceReplay {
        std::vector<float> perceptionVec;
        int actionIndex;
        float reward;
        std::vector<float> memoryContext;
    };
    std::vector<ExperienceReplay> replayBuffer;
    static constexpr size_t MAX_REPLAY_BUFFER = 100;
    float learningRate = 0.001f;
    
    // Cached last perception/action for experience replay
    std::vector<float> lastPerceptionVec;
    std::vector<float> lastMemoryContext;
    int lastActionIndex = -1;
    
    // Helper methods
    std::vector<float> perceptionToVector(const Perception& perception) const;
    std::vector<float> getMemoryContext() const;
    Action actionFromProbabilities(const std::vector<float>& probs, const Perception& perception);
    void updateMemoryBuffer(const Perception& perception, Tick currentTick);
    void computeMemoryAttention(const std::vector<float>& queryVec);
    void decayMemories(Tick currentTick);
    void triggerProustianRecall(const std::vector<float>& currentPerception);
    
    // ONNX inference
    bool loadModel(const std::string& modelPath);
    std::vector<float> runInference(const std::vector<float>& perceptionVec, 
                                     const std::vector<float>& memoryContext);
    
    // Online learning (simplified Hebbian-style update)
    void applyOnlineUpdate(const ExperienceReplay& experience);
    float computeReward(const Outcome& outcome) const;
};

} // namespace pw
