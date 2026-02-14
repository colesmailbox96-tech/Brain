#pragma once

#include "engine/Math.h"
#include <vector>

namespace pw {

class World;

class Pathfinder {
public:
    struct Node {
        int x, y;
        float g, h, f;
        Node* parent = nullptr;
        
        Node(int x_, int y_) : x(x_), y(y_), g(0), h(0), f(0) {}
    };
    
    static std::vector<Vec2> findPath(const World& world, Vec2 start, Vec2 goal, int maxSteps = 1000);
    
private:
    static float heuristic(int x1, int y1, int x2, int y2);
};

} // namespace pw
