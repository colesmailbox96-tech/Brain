#pragma once

#include <cmath>

namespace pw {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }

    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }

    Vec2 operator*(float scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    Vec2 operator/(float scalar) const {
        return Vec2(x / scalar, y / scalar);
    }

    float length() const {
        return std::sqrt(x * x + y * y);
    }

    Vec2 normalized() const {
        float len = length();
        if (len > 0.0f) {
            return Vec2(x / len, y / len);
        }
        return Vec2(0, 0);
    }

    float distance(const Vec2& other) const {
        return (*this - other).length();
    }
};

struct Rect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    Rect() = default;
    Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), w(w_), h(h_) {}

    bool contains(int px, int py) const {
        return px >= x && px < x + w && py >= y && py < y + h;
    }
};

struct Color {
    uint8_t r = 255;
    uint8_t g = 255;
    uint8_t b = 255;
    uint8_t a = 255;

    Color() = default;
    Color(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_ = 255)
        : r(r_), g(g_), b(b_), a(a_) {}

    Color withTint(const Color& tint, float strength) const {
        return Color(
            static_cast<uint8_t>(r * (1.0f - strength) + tint.r * strength),
            static_cast<uint8_t>(g * (1.0f - strength) + tint.g * strength),
            static_cast<uint8_t>(b * (1.0f - strength) + tint.b * strength),
            a
        );
    }
};

} // namespace pw
