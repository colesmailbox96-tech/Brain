#!/usr/bin/env python3
"""
Data export tool for Pixel World Simulator
Converts JSON logs to PyTorch-ready tensor datasets
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any


class DataProcessor:
    def __init__(self, log_dir: str = "data_logs"):
        self.log_dir = Path(log_dir)
        self.decisions = []
        self.events = []
        
    def load_logs(self):
        """Load all decision and event logs"""
        decisions_file = self.log_dir / "decisions.jsonl"
        events_file = self.log_dir / "events.jsonl"
        
        if decisions_file.exists():
            with open(decisions_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'tick' in data:  # Skip schema version line
                            self.decisions.append(data)
                    except json.JSONDecodeError:
                        continue
        
        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'tick' in data:
                            self.events.append(data)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.decisions)} decisions and {len(self.events)} events")
    
    def extract_features(self, perception: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from perception data"""
        features = []
        
        # Internal needs (5 values)
        needs = perception['internal_needs']
        features.extend([
            needs['hunger'],
            needs['energy'],
            needs['social'],
            needs['curiosity'],
            needs['safety']
        ])
        
        # Time of day (1 value)
        features.append(perception['time_of_day'])
        
        # Weather encoding (3 values - one-hot)
        weather = perception['weather']
        features.extend([
            1.0 if weather == 'clear' else 0.0,
            1.0 if weather == 'rain' else 0.0,
            1.0 if weather == 'storm' else 0.0
        ])
        
        # Count of nearby tiles by type (simplified)
        food_count = sum(1 for tile in perception['nearby_tiles'] if tile['type'] == 'food')
        shelter_count = sum(1 for tile in perception['nearby_tiles'] if tile['type'] == 'shelter')
        features.extend([
            min(food_count / 10.0, 1.0),  # Normalized
            min(shelter_count / 10.0, 1.0)
        ])
        
        # Count of nearby NPCs
        npc_count = len(perception['nearby_npcs'])
        features.append(min(npc_count / 10.0, 1.0))
        
        # Memory recall count
        memory_count = len(perception['memory_recalls'])
        features.append(min(memory_count / 5.0, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def encode_action(self, decision: Dict[str, Any]) -> int:
        """Encode action type as integer label"""
        action_map = {
            'idle': 0,
            'move': 1,
            'forage': 2,
            'eat': 3,
            'rest': 4,
            'explore': 5,
            'socialize': 6,
            'build_shelter': 7,
            'seek_shelter': 8
        }
        return action_map.get(decision['type'], 0)
    
    def create_dataset(self, output_dir: str = "training_data"):
        """Create numpy arrays for PyTorch training"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        features_list = []
        labels_list = []
        
        for decision in self.decisions:
            features = self.extract_features(decision['perception'])
            label = self.encode_action(decision['decision'])
            
            features_list.append(features)
            labels_list.append(label)
        
        # Convert to numpy arrays
        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        
        # Save as numpy files
        np.save(output_path / 'features.npy', X)
        np.save(output_path / 'labels.npy', y)
        
        # Save metadata
        metadata = {
            'num_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': 9,
            'feature_names': [
                'hunger', 'energy', 'social', 'curiosity', 'safety',
                'time_of_day',
                'weather_clear', 'weather_rain', 'weather_storm',
                'nearby_food', 'nearby_shelter', 'nearby_npcs', 'memory_count'
            ],
            'class_names': [
                'idle', 'move', 'forage', 'eat', 'rest', 'explore',
                'socialize', 'build_shelter', 'seek_shelter'
            ]
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset created:")
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Saved to: {output_path}")
        
        return X, y
    
    def print_statistics(self):
        """Print dataset statistics"""
        if not self.decisions:
            print("No decisions to analyze")
            return
        
        action_counts = {}
        for decision in self.decisions:
            action = decision['decision']['type']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print("\nAction distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.decisions)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        
        # Average needs
        avg_hunger = np.mean([d['perception']['internal_needs']['hunger'] for d in self.decisions])
        avg_energy = np.mean([d['perception']['internal_needs']['energy'] for d in self.decisions])
        avg_social = np.mean([d['perception']['internal_needs']['social'] for d in self.decisions])
        
        print(f"\nAverage needs:")
        print(f"  Hunger: {avg_hunger:.3f}")
        print(f"  Energy: {avg_energy:.3f}")
        print(f"  Social: {avg_social:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Process Pixel World Simulator logs')
    parser.add_argument('--log-dir', default='data_logs', help='Directory containing log files')
    parser.add_argument('--output-dir', default='training_data', help='Output directory for processed data')
    parser.add_argument('--stats', action='store_true', help='Print statistics only')
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.log_dir)
    processor.load_logs()
    processor.print_statistics()
    
    if not args.stats:
        processor.create_dataset(args.output_dir)


if __name__ == '__main__':
    main()
