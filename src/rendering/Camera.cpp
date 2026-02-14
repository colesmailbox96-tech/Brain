#include "Camera.h"

namespace pw {

Camera::Camera(int viewWidth, int viewHeight)
    : position(0, 0), viewWidth(viewWidth), viewHeight(viewHeight) {}

Vec2 Camera::worldToScreen(Vec2 worldPos) const {
    return Vec2(
        (worldPos.x - position.x) * zoom + viewWidth / 2.0f,
        (worldPos.y - position.y) * zoom + viewHeight / 2.0f
    );
}

Vec2 Camera::screenToWorld(Vec2 screenPos) const {
    return Vec2(
        (screenPos.x - viewWidth / 2.0f) / zoom + position.x,
        (screenPos.y - viewHeight / 2.0f) / zoom + position.y
    );
}

} // namespace pw
