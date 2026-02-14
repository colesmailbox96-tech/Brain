#include "ai/social/SocialIntelligence.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace pw {

// RelationshipEmbedding implementation
RelationshipEmbedding::RelationshipEmbedding(EntityId id) 
    : npcId(id) {
    // Initialize with small random values
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    embedding.resize(EMBEDDING_DIM);
    for (float& val : embedding) {
        val = dist(gen);
    }
    
    updateDerivedMetrics();
}

void RelationshipEmbedding::updateDerivedMetrics() {
    // Derive trust and affinity from embedding dimensions
    // Use first few dimensions for interpretable metrics
    if (embedding.size() >= 2) {
        trust = std::tanh(embedding[0]);  // Maps to [-1, 1]
        affinity = std::tanh(embedding[1]);
    }
}

float RelationshipEmbedding::similarity(const RelationshipEmbedding& a, 
                                       const RelationshipEmbedding& b) {
    // Cosine similarity
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    size_t minSize = std::min(a.embedding.size(), b.embedding.size());
    for (size_t i = 0; i < minSize; ++i) {
        dotProduct += a.embedding[i] * b.embedding[i];
        normA += a.embedding[i] * a.embedding[i];
        normB += b.embedding[i] * b.embedding[i];
    }
    
    normA = std::sqrt(normA);
    normB = std::sqrt(normB);
    
    if (normA < 1e-6f || normB < 1e-6f) return 0.0f;
    
    return dotProduct / (normA * normB);
}

// SocialIntelligence implementation
SocialIntelligence::SocialIntelligence(EntityId ownerId) 
    : ownerId(ownerId) {
}

void SocialIntelligence::recordInteraction(EntityId otherNpc, 
                                           const std::string& interactionType,
                                           float valence, Tick currentTick) {
    // Get or create relationship
    auto it = relationships.find(otherNpc);
    if (it == relationships.end()) {
        relationships.emplace(otherNpc, RelationshipEmbedding(otherNpc));
        it = relationships.find(otherNpc);
    }
    
    // Update the relationship
    updateEmbedding(it->second, interactionType, valence);
    it->second.lastInteraction = currentTick;
}

const RelationshipEmbedding* SocialIntelligence::getRelationship(EntityId npcId) const {
    auto it = relationships.find(npcId);
    if (it != relationships.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<EntityId> SocialIntelligence::findSimilarNpcs(float threshold) const {
    std::vector<EntityId> similar;
    
    // Compare all pairs of relationships
    std::vector<std::pair<EntityId, const RelationshipEmbedding*>> rels;
    for (const auto& [id, rel] : relationships) {
        rels.push_back({id, &rel});
    }
    
    // Find NPCs with similar relationship patterns
    for (size_t i = 0; i < rels.size(); ++i) {
        for (size_t j = i + 1; j < rels.size(); ++j) {
            float sim = RelationshipEmbedding::similarity(*rels[i].second, *rels[j].second);
            if (sim >= threshold) {
                if (std::find(similar.begin(), similar.end(), rels[i].first) == similar.end()) {
                    similar.push_back(rels[i].first);
                }
                if (std::find(similar.begin(), similar.end(), rels[j].first) == similar.end()) {
                    similar.push_back(rels[j].first);
                }
            }
        }
    }
    
    return similar;
}

EntityId SocialIntelligence::getClosestAlly() const {
    EntityId bestId = 0;
    float bestAffinity = -1.0f;
    
    for (const auto& [id, rel] : relationships) {
        if (rel.affinity > bestAffinity) {
            bestAffinity = rel.affinity;
            bestId = id;
        }
    }
    
    return bestId;
}

EntityId SocialIntelligence::getStrongestRival() const {
    EntityId worstId = 0;
    float worstAffinity = 1.0f;
    
    for (const auto& [id, rel] : relationships) {
        if (rel.affinity < worstAffinity) {
            worstAffinity = rel.affinity;
            worstId = id;
        }
    }
    
    return worstId;
}

void SocialIntelligence::decayRelationships(Tick currentTick) {
    for (auto& [id, rel] : relationships) {
        Tick timeSinceInteraction = currentTick - rel.lastInteraction;
        
        // Decay towards neutral (0) over time
        if (timeSinceInteraction > 1000) {  // After 1000 ticks
            float decayRate = 0.001f;
            for (float& val : rel.embedding) {
                val *= (1.0f - decayRate);
            }
            rel.updateDerivedMetrics();
        }
    }
}

void SocialIntelligence::updateEmbedding(RelationshipEmbedding& rel,
                                        const std::string& interactionType,
                                        float valence) {
    // Learning rate
    float alpha = 0.1f;
    
    // Update embedding based on interaction type and valence
    // This is a simplified learned update - in full version would use actual gradients
    
    if (interactionType == "cooperative" || interactionType == "share_food") {
        // Positive interactions increase trust and affinity dimensions
        rel.embedding[0] += alpha * valence * 0.5f;  // Trust
        rel.embedding[1] += alpha * valence * 0.3f;  // Affinity
    } else if (interactionType == "competitive" || interactionType == "conflict") {
        // Negative interactions decrease trust and affinity
        rel.embedding[0] -= alpha * std::abs(valence) * 0.4f;
        rel.embedding[1] -= alpha * std::abs(valence) * 0.6f;
    } else if (interactionType == "neutral" || interactionType == "observe") {
        // Neutral interactions slightly increase familiarity
        rel.embedding[2] += alpha * 0.1f;  // Familiarity dimension
    }
    
    // Add some noise to other dimensions (exploration)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.01f);
    
    for (size_t i = 3; i < rel.embedding.size(); ++i) {
        rel.embedding[i] += noise(gen);
    }
    
    // Clamp values to reasonable range
    for (float& val : rel.embedding) {
        val = std::max(-2.0f, std::min(2.0f, val));
    }
    
    rel.updateDerivedMetrics();
}

} // namespace pw
