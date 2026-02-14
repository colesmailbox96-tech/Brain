#include "Pathfinder.h"
#include "world/World.h"
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace pw {

struct NodeCompare {
    bool operator()(const Pathfinder::Node* a, const Pathfinder::Node* b) const {
        return a->f > b->f;
    }
};

float Pathfinder::heuristic(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

std::vector<Vec2> Pathfinder::findPath(const World& world, Vec2 start, Vec2 goal, int maxSteps) {
    int startX = static_cast<int>(start.x);
    int startY = static_cast<int>(start.y);
    int goalX = static_cast<int>(goal.x);
    int goalY = static_cast<int>(goal.y);
    
    // Check if goal is walkable
    if (!world.isWalkable(goalX, goalY)) {
        return {};
    }
    
    std::priority_queue<Node*, std::vector<Node*>, NodeCompare> openSet;
    std::unordered_map<int, std::unique_ptr<Node>> allNodes;
    std::unordered_set<int> closedSet;
    
    auto getKey = [](int x, int y) { return y * 10000 + x; };
    
    auto startNode = std::make_unique<Node>(startX, startY);
    startNode->g = 0;
    startNode->h = heuristic(startX, startY, goalX, goalY);
    startNode->f = startNode->h;
    
    Node* startPtr = startNode.get();
    allNodes[getKey(startX, startY)] = std::move(startNode);
    openSet.push(startPtr);
    
    int steps = 0;
    constexpr int dx[] = {0, 1, 0, -1, 1, 1, -1, -1};
    constexpr int dy[] = {-1, 0, 1, 0, -1, 1, 1, -1};
    
    while (!openSet.empty() && steps < maxSteps) {
        steps++;
        
        Node* current = openSet.top();
        openSet.pop();
        
        int currentKey = getKey(current->x, current->y);
        if (closedSet.count(currentKey)) {
            continue;
        }
        closedSet.insert(currentKey);
        
        // Goal reached
        if (current->x == goalX && current->y == goalY) {
            std::vector<Vec2> path;
            Node* node = current;
            while (node != nullptr) {
                path.push_back(Vec2(static_cast<float>(node->x), static_cast<float>(node->y)));
                node = node->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        // Check neighbors
        for (int i = 0; i < 8; i++) {
            int nx = current->x + dx[i];
            int ny = current->y + dy[i];
            int neighborKey = getKey(nx, ny);
            
            if (closedSet.count(neighborKey)) {
                continue;
            }
            
            if (!world.isWalkable(nx, ny)) {
                continue;
            }
            
            float moveCost = (i < 4) ? 1.0f : 1.414f; // Diagonal cost
            float newG = current->g + moveCost;
            
            Node* neighbor = nullptr;
            auto it = allNodes.find(neighborKey);
            if (it == allNodes.end()) {
                auto newNode = std::make_unique<Node>(nx, ny);
                neighbor = newNode.get();
                allNodes[neighborKey] = std::move(newNode);
            } else {
                neighbor = it->second.get();
                if (newG >= neighbor->g && neighbor->g > 0) {
                    continue;
                }
            }
            
            neighbor->g = newG;
            neighbor->h = heuristic(nx, ny, goalX, goalY);
            neighbor->f = neighbor->g + neighbor->h;
            neighbor->parent = current;
            openSet.push(neighbor);
        }
    }
    
    return {}; // No path found
}

} // namespace pw
