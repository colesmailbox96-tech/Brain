#include "engine/GameEngine.h"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    std::cout << "=== Pixel World Simulator - Milestone 1 ===" << std::endl;
    std::cout << "The Living Terrarium" << std::endl;
    std::cout << std::endl;
    
    pw::GameEngine engine;
    
    // Check for headless mode
    bool headless = false;
    int headlessTicks = 10000;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) {
            headless = true;
            if (i + 1 < argc) {
                headlessTicks = std::atoi(argv[i + 1]);
            }
        }
    }
    
    if (headless) {
        std::cout << "Running in headless mode for " << headlessTicks << " ticks..." << std::endl;
        engine.runHeadless(headlessTicks);
    } else {
        std::cout << "Starting visual simulation..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  WASD / Arrow Keys - Move camera" << std::endl;
        std::cout << "  +/- - Zoom in/out" << std::endl;
        std::cout << "  F3 - Toggle debug overlay" << std::endl;
        std::cout << std::endl;
        engine.run();
    }
    
    std::cout << "Simulation complete. Check data_logs/ for training data." << std::endl;
    
    return 0;
}
