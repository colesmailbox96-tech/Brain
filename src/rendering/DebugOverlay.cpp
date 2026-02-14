#include "rendering/DebugOverlay.h"
#include "ai/neural/NeuralBrain.h"
#include "ai/social/SocialIntelligence.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace pw {

DebugOverlay::DebugOverlay(Renderer& renderer) : renderer(renderer) {
}

void DebugOverlay::renderNPCDebug(const NPC& npc, int screenX, int screenY) {
    // Panel background
    renderer.drawRect(Rect{screenX, screenY, 400, 500}, Color{0, 0, 0, 200}, true);
    
    int yOffset = screenY + 10;
    const int lineHeight = 15;
    const int panelX = screenX + 10;
    
    // Title
    std::stringstream title;
    title << "NPC " << npc.getId();
    if (npc.isNeuralBrain()) {
        title << " [NEURAL]";
    } else {
        title << " [BEHAVIOR TREE]";
    }
    drawText(title.str(), panelX, yOffset, {255, 255, 100});
    yOffset += lineHeight * 2;
    
    // Basic info
    drawText("Position: " + std::to_string(static_cast<int>(npc.getPosition().x)) + ", " + 
             std::to_string(static_cast<int>(npc.getPosition().y)), panelX, yOffset);
    yOffset += lineHeight;
    
    // Needs
    const Needs& needs = npc.getNeeds();
    drawText("=== NEEDS ===", panelX, yOffset, {150, 255, 150});
    yOffset += lineHeight;
    
    drawBar(panelX, yOffset, 200, 10, needs.hunger, {255, 100, 100});
    drawText("Hunger", panelX + 210, yOffset);
    yOffset += lineHeight;
    
    drawBar(panelX, yOffset, 200, 10, needs.energy, {100, 100, 255});
    drawText("Energy", panelX + 210, yOffset);
    yOffset += lineHeight;
    
    drawBar(panelX, yOffset, 200, 10, needs.social, {100, 255, 100});
    drawText("Social", panelX + 210, yOffset);
    yOffset += lineHeight;
    
    yOffset += 5;
    
    // Neural brain specific
    if (npc.isNeuralBrain()) {
        auto* neuralBrain = dynamic_cast<const NeuralBrain*>(npc.getBrain());
        if (neuralBrain) {
            // Emotional state
            drawText("=== EMOTION ===", panelX, yOffset, {255, 150, 255});
            yOffset += lineHeight;
            renderEmotionalState(neuralBrain->getEmotionalState(), 
                               panelX, yOffset, 200, 60);
            yOffset += 70;
            
            // Action probabilities
            drawText("=== ACTION PROBS ===", panelX, yOffset, {150, 200, 255});
            yOffset += lineHeight;
            renderActionProbabilities(neuralBrain->getLastActionProbs(),
                                    panelX, yOffset, 200, 120);
            yOffset += 130;
            
            // Memory attention
            drawText("=== MEMORY ===", panelX, yOffset, {255, 200, 150});
            yOffset += lineHeight;
            const auto& memories = neuralBrain->getMemoryBuffer();
            if (memories.empty()) {
                drawText("(no memories)", panelX, yOffset, {150, 150, 150});
                yOffset += lineHeight;
            } else {
                renderMemoryActivations(memories, panelX, yOffset, 200, 80, 5);
                yOffset += 90;
            }
        }
    }
}

void DebugOverlay::renderPerceptionVector(const std::vector<float>& perception,
                                         int x, int y, int width, int height) {
    // Visualize perception as a bar chart
    if (perception.empty()) return;
    
    int barWidth = width / static_cast<int>(perception.size());
    for (size_t i = 0; i < perception.size(); ++i) {
        int barX = x + static_cast<int>(i) * barWidth;
        int barHeight = static_cast<int>(perception[i] * height);
        
        Color barColor = {100, 150, 255};
        renderer.drawRect(Rect{barX, y + height - barHeight, barWidth - 1, barHeight},
                         barColor, true);
    }
}

void DebugOverlay::renderMemoryActivations(const std::vector<EpisodicMemory>& memories,
                                          int x, int y, int width, int height, int maxShow) {
    // Sort by attention weight
    std::vector<const EpisodicMemory*> sorted;
    for (const auto& mem : memories) {
        sorted.push_back(&mem);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const EpisodicMemory* a, const EpisodicMemory* b) {
                  return a->attentionWeight > b->attentionWeight;
              });
    
    // Show top memories
    int lineHeight = height / std::max(1, maxShow);
    for (int i = 0; i < std::min(maxShow, static_cast<int>(sorted.size())); ++i) {
        const auto* mem = sorted[i];
        int yPos = y + i * lineHeight;
        
        // Draw attention weight bar
        int barWidth = static_cast<int>(mem->attentionWeight * width);
        Color memColor = {150, 100, 255};
        renderer.drawRect(Rect{x, yPos, barWidth, lineHeight - 2}, memColor, true);
        
        // Draw memory type
        std::stringstream ss;
        ss << mem->memory.type << " " << std::fixed << std::setprecision(2) 
           << mem->attentionWeight;
        drawText(ss.str(), x + 5, yPos + 2, {255, 255, 255});
    }
}

void DebugOverlay::renderEmotionalState(const EmotionalState& emotion,
                                       int x, int y, int width, int height) {
    const int barHeight = 15;
    
    // Valence bar (-1 to +1)
    float valenceNorm = (emotion.valence + 1.0f) / 2.0f;  // Normalize to [0, 1]
    Color valenceColor = emotion.valence > 0 ? Color{100, 255, 100} : Color{255, 100, 100};
    drawBar(x, y, width, barHeight, valenceNorm, valenceColor);
    drawText("Valence", x + width + 5, y);
    
    // Arousal bar (-1 to +1)
    float arousalNorm = (emotion.arousal + 1.0f) / 2.0f;
    drawBar(x, y + barHeight + 5, width, barHeight, arousalNorm, {255, 200, 100});
    drawText("Arousal", x + width + 5, y + barHeight + 5);
    
    // Dominance bar (-1 to +1)
    float dominanceNorm = (emotion.dominance + 1.0f) / 2.0f;
    drawBar(x, y + (barHeight + 5) * 2, width, barHeight, dominanceNorm, {100, 200, 255});
    drawText("Dominance", x + width + 5, y + (barHeight + 5) * 2);
}

void DebugOverlay::renderActionProbabilities(const std::vector<float>& probs,
                                            int x, int y, int width, int height) {
    if (probs.size() != 9) return;  // Should be 9 action types
    
    const int barHeight = height / 9;
    const char* actionNames[] = {
        "Idle", "Move", "Forage", "Eat", "Rest", 
        "Explore", "Socialize", "Build", "Shelter"
    };
    
    for (size_t i = 0; i < probs.size(); ++i) {
        int yPos = y + static_cast<int>(i) * barHeight;
        int barWidth = static_cast<int>(probs[i] * width);
        
        Color barColor = {100, 150, 255};
        // Highlight highest probability
        if (probs[i] == *std::max_element(probs.begin(), probs.end())) {
            barColor = {255, 200, 100};
        }
        
        renderer.drawRect(Rect{x, yPos, barWidth, barHeight - 2}, barColor, true);
        
        // Draw action name and percentage
        std::stringstream ss;
        ss << actionNames[i] << " " << std::fixed << std::setprecision(1) 
           << (probs[i] * 100.0f) << "%";
        drawText(ss.str(), x + 5, yPos + 2, {255, 255, 255});
    }
}

void DebugOverlay::renderSocialEmbeddings(
    const std::map<EntityId, RelationshipEmbedding>& relationships,
    int x, int y, int width, int height, int maxShow) {
    
    // Sort by affinity
    std::vector<std::pair<EntityId, const RelationshipEmbedding*>> sorted;
    for (const auto& [id, rel] : relationships) {
        sorted.push_back({id, &rel});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return std::abs(a.second->affinity) > std::abs(b.second->affinity);
              });
    
    int lineHeight = height / std::max(1, maxShow);
    for (int i = 0; i < std::min(maxShow, static_cast<int>(sorted.size())); ++i) {
        const auto& [npcId, rel] = sorted[i];
        int yPos = y + i * lineHeight;
        
        // Draw affinity bar
        float affinityNorm = (rel->affinity + 1.0f) / 2.0f;
        Color relColor = rel->affinity > 0 ? Color{100, 255, 100} : Color{255, 100, 100};
        
        int barWidth = static_cast<int>(affinityNorm * width);
        renderer.drawRect(Rect{x, yPos, barWidth, lineHeight - 2}, relColor, true);
        
        // Draw NPC ID and metrics
        std::stringstream ss;
        ss << "NPC" << npcId << " A:" << std::fixed << std::setprecision(1) 
           << rel->affinity << " T:" << rel->trust;
        drawText(ss.str(), x + 5, yPos + 2, {255, 255, 255});
    }
}

void DebugOverlay::drawText(const std::string& text, int x, int y, const Color& color) {
    // TODO: Actual text rendering requires SDL_ttf or bitmap font system
    // This is a placeholder showing where text would be rendered
    // For production, integrate SDL_ttf:
    //   TTF_Font* font = TTF_OpenFont("font.ttf", 12);
    //   SDL_Surface* surface = TTF_RenderText_Solid(font, text.c_str(), sdlColor);
    //   // Convert surface to texture and render
    
    renderer.setDrawColor(color);
    renderer.drawRect(Rect{x, y, static_cast<int>(text.length()) * 6, 10}, color, false);
}

void DebugOverlay::drawBar(int x, int y, int width, int height, float value,
                          const Color& color, const Color& bgColor) {
    // Background
    renderer.drawRect(Rect{x, y, width, height}, bgColor, true);
    
    // Foreground (value bar)
    int filledWidth = static_cast<int>(value * width);
    renderer.drawRect(Rect{x, y, filledWidth, height}, color, true);
    
    // Border
    renderer.drawRect(Rect{x, y, width, height}, {200, 200, 200}, false);
}

std::string DebugOverlay::getActionName(ActionType type) const {
    switch (type) {
        case ActionType::Idle: return "Idle";
        case ActionType::Move: return "Move";
        case ActionType::Forage: return "Forage";
        case ActionType::Eat: return "Eat";
        case ActionType::Rest: return "Rest";
        case ActionType::Explore: return "Explore";
        case ActionType::Socialize: return "Socialize";
        case ActionType::BuildShelter: return "BuildShelter";
        case ActionType::SeekShelter: return "SeekShelter";
        default: return "Unknown";
    }
}

} // namespace pw
