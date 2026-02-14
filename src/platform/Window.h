#pragma once

#include "engine/Math.h"
#include <SDL2/SDL.h>
#include <string>
#include <memory>

namespace pw {

class Window {
public:
    Window(const std::string& title, int width, int height);
    ~Window();
    
    bool isOpen() const { return !shouldClose; }
    void close() { shouldClose = true; }
    
    SDL_Renderer* getRenderer() { return renderer; }
    
    void clear(Color color);
    void present();
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
    // Virtual resolution support
    void setVirtualResolution(int vWidth, int vHeight);
    void applyVirtualScale();

private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    bool shouldClose = false;
    int width, height;
    int virtualWidth = 0, virtualHeight = 0;
    float scaleX = 1.0f, scaleY = 1.0f;
    int offsetX = 0, offsetY = 0;
};

} // namespace pw
