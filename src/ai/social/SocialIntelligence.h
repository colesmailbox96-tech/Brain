#pragma once

#include "engine/Types.h"
#include "engine/Math.h"
#include <map>
#include <vector>
#include <string>

namespace pw {

// Relationship embedding - learned representation of NPC-to-NPC relationship
struct RelationshipEmbedding {
    EntityId npcId;
    std::vector<float> embedding;  // Learned embedding vector
    float trust = 0.0f;            // Derived from embedding
    float affinity = 0.0f;         // Derived from embedding
    Tick lastInteraction = 0;
    
    static constexpr size_t EMBEDDING_DIM = 16;
    
    RelationshipEmbedding(EntityId id);
    
    // Compute derived metrics from embedding
    void updateDerivedMetrics();
    
    // Measure similarity between two embeddings
    static float similarity(const RelationshipEmbedding& a, const RelationshipEmbedding& b);
};

class SocialIntelligence {
public:
    SocialIntelligence(EntityId ownerId);
    
    // Update relationship based on interaction outcome
    void recordInteraction(EntityId otherNpc, const std::string& interactionType, 
                          float valence, Tick currentTick);
    
    // Get relationship with specific NPC
    const RelationshipEmbedding* getRelationship(EntityId npcId) const;
    
    // Get all relationships
    const std::map<EntityId, RelationshipEmbedding>& getAllRelationships() const { 
        return relationships; 
    }
    
    // Find NPCs with similar relationships (emergent groups)
    std::vector<EntityId> findSimilarNpcs(float threshold = 0.7f) const;
    
    // Get strongest positive/negative relationships
    EntityId getClosestAlly() const;
    EntityId getStrongestRival() const;
    
    // Decay relationships over time
    void decayRelationships(Tick currentTick);

private:
    EntityId ownerId;
    std::map<EntityId, RelationshipEmbedding> relationships;
    
    // Update embedding based on interaction
    void updateEmbedding(RelationshipEmbedding& rel, const std::string& interactionType, 
                        float valence);
};

} // namespace pw
