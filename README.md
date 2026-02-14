# Brain - Pixel World Simulator

## Milestone 1 — THE LIVING TERRARIUM

A complete 2D pixel-art world simulation in C++17 with SDL2. Watch NPCs living, learning, and generating the data that will teach them to think.

## Features

- **Procedural World Generation**: Organic terrain using simplex noise (grass, dirt, water, stone, trees, berry bushes, caves)
- **Day/Night Cycle**: Smooth lighting transitions affecting NPC behavior
- **Weather System**: Rain and storms that NPCs respond to
- **Intelligent NPCs**: 15 NPCs with sophisticated behavior trees
  - Needs system: hunger, energy, social, curiosity, safety
  - A* pathfinding
  - Memory system for learning locations
  - Mood/emotion reflected in appearance and movement
  - Behaviors: foraging, resting, socializing, exploring, building, seeking shelter
- **Data Logging**: Complete perception-decision-outcome logging for ML training
- **Headless Mode**: Run simulation at any speed for rapid data generation

## Build & Run

### Prerequisites

- CMake 3.15+
- C++17 compiler
- SDL2 development libraries

On Ubuntu/Debian:
```bash
sudo apt-get install cmake build-essential libsdl2-dev
```

On macOS:
```bash
brew install cmake sdl2
```

### Compilation

```bash
cmake -B build
cmake --build build
```

### Run Visual Simulation

```bash
./build/pixel_world_sim
```

**Controls:**
- WASD / Arrow Keys - Move camera
- +/- - Zoom in/out
- F3 - Toggle debug overlay

### Run Headless (Data Generation)

```bash
./build/pixel_world_sim --headless 10000
```

This runs 10,000 simulation ticks and generates training data in `data_logs/`.

## Data Export for ML Training

### Setup Python Environment

```bash
cd tools
pip install -r requirements.txt
```

### Export Training Data

```bash
python tools/export_training_data.py --log-dir data_logs --output-dir training_data
```

This creates PyTorch-ready numpy arrays:
- `training_data/features.npy` - 13D feature vectors (needs, environment, memory)
- `training_data/labels.npy` - Action labels (9 classes)
- `training_data/metadata.json` - Schema information

### View Statistics

```bash
python tools/export_training_data.py --stats
```

## Architecture

```
/src
  /engine        — Core loop, timing, types
  /platform      — SDL2 window abstraction
  /world         — Terrain generation, tiles, weather
  /entities      — NPC system with needs and memory
  /ai
    /interface   — IBrain abstract interface
    /behavior    — Behavior trees, pathfinding
    /memory      — Episodic memory with significance scoring
  /rendering     — Procedural sprites, camera
  /input         — Input abstraction
  /data          — Perception-decision-outcome logging
/tools           — Python data export scripts
```

## Design Principles

1. **Platform Independence**: Game logic is fully decoupled from rendering
2. **Fixed Timestep Simulation**: Deterministic, can run headless at any speed
3. **Modular AI**: IBrain interface allows easy brain swapping per NPC
4. **Data-First**: Logging is a first-class system, not debugging
5. **Mobile-Ready**: Touch-first input design (keyboard/mouse mapped on top)

## Next Steps

This foundation supports:
- Neural network training on NPC decision data
- ONNX model export for inference
- Mobile deployment (iOS/Android)
- Reinforcement learning integration
- Multi-agent learning systems

## License

MIT