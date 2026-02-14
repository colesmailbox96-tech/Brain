#pragma once

#include <SDL2/SDL.h>
#include <unordered_map>

namespace pw {

enum class InputAction {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    ZoomIn,
    ZoomOut,
    Select,
    Cancel,
    ToggleDebug,
    CycleNPC
};

class InputManager {
public:
    void processEvent(const SDL_Event& event);
    void update();
    
    bool isActionPressed(InputAction action) const;
    bool isActionJustPressed(InputAction action) const;
    
    void getMousePosition(int& x, int& y) const { x = mouseX; y = mouseY; }

private:
    std::unordered_map<InputAction, bool> currentState;
    std::unordered_map<InputAction, bool> previousState;
    int mouseX = 0, mouseY = 0;
    
    void mapKeyToAction(SDL_Keycode key, InputAction action, bool pressed);
};

} // namespace pw
