#pragma once

#include "engine/Math.h"
#include <SDL2/SDL.h>

namespace pw {

class Renderer {
public:
    explicit Renderer(SDL_Renderer* renderer);
    
    void drawRect(const Rect& rect, const Color& color, bool filled = true);
    void drawCircle(int centerX, int centerY, int radius, const Color& color, bool filled = true);
    void drawLine(int x1, int y1, int x2, int y2, const Color& color);
    
    void setDrawColor(const Color& color);
    SDL_Renderer* getSDLRenderer() { return renderer; }

private:
    SDL_Renderer* renderer;
};

} // namespace pw
