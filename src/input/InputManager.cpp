#include "InputManager.h"

namespace pw {

void InputManager::processEvent(const SDL_Event& event) {
    if (event.type == SDL_KEYDOWN) {
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveUp, true);
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveDown, true);
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveLeft, true);
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveRight, true);
        mapKeyToAction(event.key.keysym.sym, InputAction::ZoomIn, true);
        mapKeyToAction(event.key.keysym.sym, InputAction::ZoomOut, true);
        mapKeyToAction(event.key.keysym.sym, InputAction::ToggleDebug, true);
    } else if (event.type == SDL_KEYUP) {
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveUp, false);
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveDown, false);
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveLeft, false);
        mapKeyToAction(event.key.keysym.sym, InputAction::MoveRight, false);
    } else if (event.type == SDL_MOUSEMOTION) {
        mouseX = event.motion.x;
        mouseY = event.motion.y;
    }
}

void InputManager::update() {
    previousState = currentState;
}

void InputManager::mapKeyToAction(SDL_Keycode key, InputAction action, bool pressed) {
    switch (action) {
        case InputAction::MoveUp:
            if (key == SDLK_w || key == SDLK_UP) currentState[action] = pressed;
            break;
        case InputAction::MoveDown:
            if (key == SDLK_s || key == SDLK_DOWN) currentState[action] = pressed;
            break;
        case InputAction::MoveLeft:
            if (key == SDLK_a || key == SDLK_LEFT) currentState[action] = pressed;
            break;
        case InputAction::MoveRight:
            if (key == SDLK_d || key == SDLK_RIGHT) currentState[action] = pressed;
            break;
        case InputAction::ZoomIn:
            if (key == SDLK_EQUALS || key == SDLK_PLUS) currentState[action] = pressed;
            break;
        case InputAction::ZoomOut:
            if (key == SDLK_MINUS) currentState[action] = pressed;
            break;
        case InputAction::ToggleDebug:
            if (key == SDLK_F3) currentState[action] = pressed;
            break;
        default:
            break;
    }
}

bool InputManager::isActionPressed(InputAction action) const {
    auto it = currentState.find(action);
    return it != currentState.end() && it->second;
}

bool InputManager::isActionJustPressed(InputAction action) const {
    auto curr = currentState.find(action);
    auto prev = previousState.find(action);
    
    bool isPressed = curr != currentState.end() && curr->second;
    bool wasPressed = prev != previousState.end() && prev->second;
    
    return isPressed && !wasPressed;
}

} // namespace pw
