#pragma once

#include "engine/Math.h"

namespace pw {

class Camera {
public:
    Camera(int viewWidth, int viewHeight);
    
    void setPosition(Vec2 pos) { position = pos; }
    void move(Vec2 delta) { position = position + delta; }
    
    Vec2 getPosition() const { return position; }
    float getZoom() const { return zoom; }
    void setZoom(float z) { zoom = z; }
    
    Vec2 worldToScreen(Vec2 worldPos) const;
    Vec2 screenToWorld(Vec2 screenPos) const;
    
    int getViewWidth() const { return viewWidth; }
    int getViewHeight() const { return viewHeight; }

private:
    Vec2 position;
    float zoom = 1.0f;
    int viewWidth, viewHeight;
};

} // namespace pw
