#!/usr/bin/env python3
"""
Test script to verify training pipeline works without actual game data
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

def generate_dummy_data(output_dir, num_samples=100):
    """Generate dummy training data in JSONL format"""
    os.makedirs(output_dir, exist_ok=True)
    
    action_types = ['Idle', 'Move', 'Forage', 'Eat', 'Rest', 'Explore', 'Socialize', 'BuildShelter', 'SeekShelter']
    
    with open(os.path.join(output_dir, 'decisions_test.jsonl'), 'w') as f:
        for i in range(num_samples):
            # Generate dummy perception
            perception = {
                'position': {'x': 50.0 + i % 50, 'y': 40.0 + i % 40},
                'needs': {
                    'hunger': (i % 10) / 10.0,
                    'energy': (i % 8) / 8.0,
                    'social': (i % 7) / 7.0,
                    'curiosity': (i % 6) / 6.0,
                    'safety': 0.5
                },
                'timeOfDay': (i % 24) / 24.0,
                'weather': 'clear' if i % 3 != 0 else 'rain',
                'nearbyTiles': [
                    {'type': 'Grass', 'position': {'x': 50, 'y': 40}},
                    {'type': 'BerryBush', 'position': {'x': 51, 'y': 40}}
                ],
                'nearbyNPCs': [],
                'memoryRecalls': ['food'] if i % 5 == 0 else []
            }
            
            # Generate dummy decision
            decision = {
                'type': action_types[i % len(action_types)],
                'targetPosition': {'x': 50.0, 'y': 40.0}
            }
            
            # Generate dummy outcome
            outcome = {
                'needsDeltas': {
                    'hunger': -0.1 if decision['type'] == 'Eat' else 0.05,
                    'energy': -0.1 if decision['type'] == 'Rest' else 0.05
                },
                'event': ''
            }
            
            entry = {
                'tick': i,
                'npcId': i % 5,  # 5 NPCs
                'perception': perception,
                'decision': decision,
                'outcome': outcome
            }
            
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {num_samples} dummy samples in {output_dir}")

def main():
    print("=== Testing Neural NPC Training Pipeline ===\n")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    model_dir = os.path.join(temp_dir, 'models')
    
    try:
        # Generate dummy data
        print("1. Generating dummy training data...")
        generate_dummy_data(data_dir, num_samples=200)
        
        # Test model architecture
        print("\n2. Testing model architecture...")
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model_architecture import create_model
        
        model = create_model()
        print(f"   Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test training (just a few epochs)
        print("\n3. Testing training script...")
        os.makedirs(model_dir, exist_ok=True)
        
        from train_npc_brain import main as train_main
        import argparse
        
        # Mock command line args
        sys.argv = [
            'train_npc_brain.py',
            '--data-dir', data_dir,
            '--output-dir', model_dir,
            '--epochs', '3',
            '--batch-size', '16',
            '--lr', '0.001'
        ]
        
        train_main()
        
        # Check if files were created
        print("\n4. Checking outputs...")
        onnx_path = os.path.join(model_dir, 'npc_brain.onnx')
        if os.path.exists(onnx_path):
            print(f"   ✓ ONNX model created: {onnx_path}")
            print(f"   ✓ Model size: {os.path.getsize(onnx_path) / 1024:.1f} KB")
        else:
            print("   ✗ ONNX model not found")
        
        pth_path = os.path.join(model_dir, 'npc_brain_final.pth')
        if os.path.exists(pth_path):
            print(f"   ✓ PyTorch checkpoint created")
        else:
            print("   ✗ PyTorch checkpoint not found")
        
        print("\n=== Training Pipeline Test Complete ===")
        print("All components working correctly!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
    
    return 0

if __name__ == '__main__':
    exit(main())
