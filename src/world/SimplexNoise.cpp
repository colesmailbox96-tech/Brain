#include "SimplexNoise.h"
#include <algorithm>
#include <random>

namespace pw {

SimplexNoise::SimplexNoise(uint32_t seed) {
    // Initialize permutation table
    std::array<uint8_t, 256> p;
    for (int i = 0; i < 256; i++) {
        p[i] = static_cast<uint8_t>(i);
    }
    
    // Shuffle with seed
    std::mt19937 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    
    // Duplicate for wrapping
    for (int i = 0; i < 256; i++) {
        perm[i] = p[i];
        perm[256 + i] = p[i];
    }
}

float SimplexNoise::grad(int hash, float x, float y) const {
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

float SimplexNoise::noise(float xin, float yin) const {
    float n0, n1, n2;
    
    // Skew input space to determine which simplex cell we're in
    float s = (xin + yin) * F2;
    int i = static_cast<int>(std::floor(xin + s));
    int j = static_cast<int>(std::floor(yin + s));
    
    float t = (i + j) * G2;
    float X0 = i - t;
    float Y0 = j - t;
    float x0 = xin - X0;
    float y0 = yin - Y0;
    
    // Determine which simplex we're in
    int i1, j1;
    if (x0 > y0) {
        i1 = 1;
        j1 = 0;
    } else {
        i1 = 0;
        j1 = 1;
    }
    
    float x1 = x0 - i1 + G2;
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;
    
    // Hash coordinates
    int ii = i & 255;
    int jj = j & 255;
    int gi0 = perm[ii + perm[jj]];
    int gi1 = perm[ii + i1 + perm[jj + j1]];
    int gi2 = perm[ii + 1 + perm[jj + 1]];
    
    // Calculate contributions from three corners
    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 < 0) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * grad(gi0, x0, y0);
    }
    
    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 < 0) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * grad(gi1, x1, y1);
    }
    
    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 < 0) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * grad(gi2, x2, y2);
    }
    
    // Scale to [-1, 1]
    return 70.0f * (n0 + n1 + n2);
}

float SimplexNoise::octaveNoise(float x, float y, int octaves, float persistence) const {
    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maxValue = 0.0f;
    
    for (int i = 0; i < octaves; i++) {
        total += noise(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }
    
    return total / maxValue;
}

} // namespace pw
