#include "Renderer.h"
#include <cmath>

namespace pw {

Renderer::Renderer(SDL_Renderer* renderer) : renderer(renderer) {}

void Renderer::setDrawColor(const Color& color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

void Renderer::drawRect(const Rect& rect, const Color& color, bool filled) {
    SDL_Rect sdlRect = {rect.x, rect.y, rect.w, rect.h};
    setDrawColor(color);
    
    if (filled) {
        SDL_RenderFillRect(renderer, &sdlRect);
    } else {
        SDL_RenderDrawRect(renderer, &sdlRect);
    }
}

void Renderer::drawCircle(int centerX, int centerY, int radius, const Color& color, bool filled) {
    setDrawColor(color);
    
    if (filled) {
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                if (x * x + y * y <= radius * radius) {
                    SDL_RenderDrawPoint(renderer, centerX + x, centerY + y);
                }
            }
        }
    } else {
        int x = radius;
        int y = 0;
        int err = 0;
        
        while (x >= y) {
            SDL_RenderDrawPoint(renderer, centerX + x, centerY + y);
            SDL_RenderDrawPoint(renderer, centerX + y, centerY + x);
            SDL_RenderDrawPoint(renderer, centerX - y, centerY + x);
            SDL_RenderDrawPoint(renderer, centerX - x, centerY + y);
            SDL_RenderDrawPoint(renderer, centerX - x, centerY - y);
            SDL_RenderDrawPoint(renderer, centerX - y, centerY - x);
            SDL_RenderDrawPoint(renderer, centerX + y, centerY - x);
            SDL_RenderDrawPoint(renderer, centerX + x, centerY - y);
            
            y++;
            err += 1 + 2 * y;
            if (2 * (err - x) + 1 > 0) {
                x--;
                err += 1 - 2 * x;
            }
        }
    }
}

void Renderer::drawLine(int x1, int y1, int x2, int y2, const Color& color) {
    setDrawColor(color);
    SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
}

} // namespace pw
