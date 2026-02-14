# Pixel World Simulator - Python Tools

## Installation

```bash
pip install -r requirements.txt
```

## Data Export

Convert simulation logs to PyTorch-ready datasets:

```bash
python export_training_data.py --log-dir ../data_logs --output-dir ../training_data
```

This will create:
- `training_data/features.npy` - Input features (perception data)
- `training_data/labels.npy` - Action labels
- `training_data/metadata.json` - Dataset information

## Statistics

View simulation statistics without exporting:

```bash
python export_training_data.py --stats
```

## Feature Schema

The exported features include:
1. Internal needs (5): hunger, energy, social, curiosity, safety
2. Time of day (1): 0.0-1.0
3. Weather (3): one-hot encoding (clear, rain, storm)
4. Nearby resources (2): normalized counts of food and shelter
5. Social context (1): normalized count of nearby NPCs
6. Memory (1): normalized count of significant memories

Total: 13 features per sample

## Action Labels

0. idle
1. move
2. forage
3. eat
4. rest
5. explore
6. socialize
7. build_shelter
8. seek_shelter
