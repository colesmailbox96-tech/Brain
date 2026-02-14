#pragma once

#include "rendering/Renderer.h"
#include "entities/NPC.h"
#include "ai/neural/NeuralBrain.h"
#include <vector>
#include <string>

namespace pw {

class DebugOverlay {
public:
    DebugOverlay(Renderer& renderer);
    
    // Main debug render for selected NPC
    void renderNPCDebug(const NPC& npc, int screenX, int screenY);
    
    // Individual components
    void renderPerceptionVector(const std::vector<float>& perception, 
                                int x, int y, int width, int height);
    
    void renderMemoryActivations(const std::vector<EpisodicMemory>& memories,
                                 int x, int y, int width, int height, int maxShow = 5);
    
    void renderEmotionalState(const EmotionalState& emotion,
                            int x, int y, int width, int height);
    
    void renderActionProbabilities(const std::vector<float>& probs,
                                   int x, int y, int width, int height);
    
    void renderSocialEmbeddings(const std::map<EntityId, RelationshipEmbedding>& relationships,
                               int x, int y, int width, int height, int maxShow = 5);
    
    // Text rendering helper
    void drawText(const std::string& text, int x, int y, const Color& color = {255, 255, 255});
    
    // Bar chart helper
    void drawBar(int x, int y, int width, int height, float value, 
                const Color& color, const Color& bgColor = {50, 50, 50});

private:
    Renderer& renderer;
    
    // Get action name from type
    std::string getActionName(ActionType type) const;
};

} // namespace pw
