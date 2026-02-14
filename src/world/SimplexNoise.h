#pragma once

#include <cstdint>
#include <cmath>
#include <array>

namespace pw {

// Simplex noise implementation for procedural generation
class SimplexNoise {
public:
    SimplexNoise(uint32_t seed = 0);
    
    float noise(float x, float y) const;
    float octaveNoise(float x, float y, int octaves, float persistence) const;

private:
    std::array<uint8_t, 512> perm;
    
    static constexpr float F2 = 0.5f * (1.7320508f - 1.0f); // sqrt(3) - 1) / 2
    static constexpr float G2 = (3.0f - 1.7320508f) / 6.0f; // (3 - sqrt(3)) / 6
    
    float grad(int hash, float x, float y) const;
};

} // namespace pw
