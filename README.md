# Brain - Pixel World Simulator

## Milestone 2 — THE AWAKENING

Neural network-based NPC AI built on the Milestone 1 foundation. NPCs now use learned transformer models with episodic memory, emotional intelligence, and social relationship dynamics.

### New Features

- **Neural NPCs**: 50% of NPCs use transformer-based decision networks instead of behavior trees
- **Episodic Memory**: Attention mechanism over significant past experiences influences decisions
- **Emotional Model**: 3D emotional state (valence/arousal/dominance) emerges from neural network
- **Social Intelligence**: Learned relationship embeddings enable emergent trust, rivalry, and group dynamics
- **On-Device Learning**: Reward-modulated online learning via experience replay buffer
- **State Persistence**: NPC brain state (emotions, memories, social bonds) saved/loaded as JSON
- **Enhanced Debug Overlay**: Visualize perception vectors, memory attention, emotions, action probabilities, social relationships
- **Training Pipeline**: PyTorch-based transformer training on Milestone 1 behavior tree data

## Milestone 1 — THE LIVING TERRARIUM

A complete 2D pixel-art world simulation in C++17 with SDL2. Watch NPCs living, learning, and generating the data that will teach them to think.

### Features

- **Procedural World Generation**: Organic terrain using simplex noise (grass, dirt, water, stone, trees, berry bushes, caves)
- **Day/Night Cycle**: Smooth lighting transitions affecting NPC behavior
- **Weather System**: Rain and storms that NPCs respond to
- **Intelligent NPCs**: 15 NPCs with sophisticated behavior trees OR neural brains
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
- Python 3.8+ with PyTorch (for training)

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
- Tab - Cycle through NPCs (when debug overlay is active)

### Run Headless (Data Generation)

```bash
./build/pixel_world_sim --headless 10000
```

This runs 10,000 simulation ticks and generates training data in `data_logs/`.

## Neural Network Training (Milestone 2)

### Setup Python Environment

```bash
cd tools
pip install -r requirements.txt
```

This installs PyTorch, ONNX, and other dependencies needed for training.

### Step 1: Generate Training Data

First, run the simulator in headless mode to generate behavior tree decision data:

```bash
./build/pixel_world_sim --headless 50000
```

This creates JSONL files in `data_logs/` with perception-action-outcome tuples.

### Step 2: Train Neural NPC Brain

Train the transformer model on the behavior tree data:

```bash
python tools/train_npc_brain.py --data-dir data_logs --output-dir models --epochs 50
```

This will:
- Load training data from `data_logs/`
- Train a transformer-based decision network
- Export the model to `models/npc_brain.onnx` (ONNX format for C++ inference)
- Save PyTorch checkpoints in `models/` directory

Training takes ~5-15 minutes on CPU depending on data size.

### Step 3: Fine-Tune for Personalities (Optional)

Create unique NPC personalities by fine-tuning on individual experience logs:

```bash
python tools/fine_tune_personality.py \
  --base-model models/npc_brain_best.pth \
  --npc-id 3 \
  --data-dir data_logs \
  --output-dir models \
  --epochs 20
```

This creates `models/npc_brain_3.onnx` with personality traits learned from NPC 3's experiences.

### Step 4: Run with Neural NPCs

The simulator automatically loads `models/npc_brain.onnx` if it exists. Half the NPCs will use neural brains, half will use behavior trees for comparison.

```bash
./build/pixel_world_sim
```

Press **F3** to toggle debug overlay, then **Tab** to cycle through NPCs and see:
- Neural vs Behavior Tree brain indicators
- Emotional state visualization (valence, arousal, dominance)
- Memory attention weights
- Action probability distributions

## Data Export for ML Training (Milestone 1)

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

## State Persistence

Neural NPC states are automatically saved and loaded between sessions:

```bash
# States are saved to npc_states/ when simulation ends
ls npc_states/
# npc_0_state.json  npc_2_state.json  npc_4_state.json ...
```

Each state file contains:
- **Emotional state**: Valence, arousal, dominance values
- **Memory buffer**: Episodic memories with embeddings and attention weights
- **Social relationships**: Relationship embeddings with trust/affinity metrics
- **Experience replay size**: Number of stored learning experiences

States are saved as human-readable JSON, making it easy to inspect and modify NPC personalities.

## Architecture

```
/src
  /engine        — Core loop, timing, types
  /platform      — SDL2 window abstraction
  /world         — Terrain generation, tiles, weather
  /entities      — NPC system with needs and memory
  /ai
    /interface   — IBrain abstract interface
    /behavior    — Behavior trees, pathfinding (Milestone 1)
    /neural      — Neural brain with ONNX Runtime, online learning (Milestone 2)
    /memory      — Episodic memory with significance scoring
    /social      — Relationship embeddings, social intelligence (Milestone 2)
  /rendering     — Procedural sprites, camera, debug overlay
  /input         — Input abstraction
  /data          — Perception-decision-outcome logging
/tools           — Python training pipeline
  model_architecture.py  — Transformer definition
  train_npc_brain.py     — Main training script
  fine_tune_personality.py — Per-NPC fine-tuning
```

## Design Principles

1. **Platform Independence**: Game logic is fully decoupled from rendering
2. **Fixed Timestep Simulation**: Deterministic, can run headless at any speed
3. **Modular AI**: IBrain interface allows easy brain swapping per NPC
4. **Data-First**: Logging is a first-class system, not debugging
5. **Mobile-Ready**: Touch-first input design (keyboard/mouse mapped on top)
6. **Learned Behavior**: Neural NPCs bootstrap from behavior trees, then adapt

## What Makes This Special

### The Awakening (Milestone 2)

Traditional game AI uses hand-coded behavior trees. Neural NPCs use **learned decision networks** that:

1. **Remember**: Transformer attention over episodic memories — NPCs recall relevant past experiences when deciding
2. **Feel**: Emotional state emerges from the network, not hand-coded rules
3. **Relate**: Social relationship embeddings enable emergent trust, rivalry, and group dynamics
4. **Adapt**: Can continue learning during gameplay (foundation for online learning)

The same `IBrain` interface means you can directly compare behavior tree and neural NPCs side-by-side.

### Example Emergent Behaviors

- NPC attacked near a river develops fear of water (memory + emotion)
- Two NPCs that repeatedly share food develop positive relationship embeddings
- High-arousal NPC explores more aggressively (emotion modulates action probabilities)
- "Proustian recall": old memory suddenly resurfaces, causing behavior shift

## Future Enhancements

- **Reinforcement Learning**: Train with PPO or similar for goal-directed behavior
- **Multi-Agent Coordination**: Emergent cooperation from relationship embeddings
- **Mobile Deployment**: Run neural NPCs on iOS/Android with optimized ONNX Runtime

## License

MIT