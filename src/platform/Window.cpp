#include "Window.h"
#include <iostream>
#include <algorithm>

namespace pw {

Window::Window(const std::string& title, int width, int height)
    : width(width), height(height) {
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return;
    }
    
    window = SDL_CreateWindow(
        title.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width,
        height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        return;
    }
    
    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    
    if (!renderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        return;
    }
    
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
}

Window::~Window() {
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
}

void Window::clear(Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    SDL_RenderClear(renderer);
}

void Window::present() {
    SDL_RenderPresent(renderer);
}

void Window::setVirtualResolution(int vWidth, int vHeight) {
    virtualWidth = vWidth;
    virtualHeight = vHeight;
    
    // Calculate scale to fit virtual resolution
    float aspectRatioVirtual = static_cast<float>(vWidth) / vHeight;
    float aspectRatioWindow = static_cast<float>(width) / height;
    
    if (aspectRatioWindow > aspectRatioVirtual) {
        // Window is wider - letterbox on sides
        scaleY = static_cast<float>(height) / vHeight;
        scaleX = scaleY;
        int scaledWidth = static_cast<int>(vWidth * scaleX);
        offsetX = (width - scaledWidth) / 2;
        offsetY = 0;
    } else {
        // Window is taller - letterbox on top/bottom
        scaleX = static_cast<float>(width) / vWidth;
        scaleY = scaleX;
        int scaledHeight = static_cast<int>(vHeight * scaleY);
        offsetX = 0;
        offsetY = (height - scaledHeight) / 2;
    }
}

void Window::applyVirtualScale() {
    if (virtualWidth > 0 && virtualHeight > 0) {
        SDL_RenderSetScale(renderer, scaleX, scaleY);
        SDL_RenderSetViewport(renderer, nullptr);
        
        // Set viewport with letterboxing
        SDL_Rect viewport = {
            static_cast<int>(offsetX / scaleX),
            static_cast<int>(offsetY / scaleY),
            virtualWidth,
            virtualHeight
        };
        SDL_RenderSetViewport(renderer, &viewport);
    }
}

} // namespace pw
