"""
Train NPC Brain - Bootstrap neural NPC behavior from behavior tree data
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_architecture import create_model


class NPCDataset(Dataset):
    """Dataset for NPC perception-action-outcome sequences"""
    
    def __init__(self, data_dir, perception_dim=20, memory_seq_len=50, memory_dim=32):
        self.perception_dim = perception_dim
        self.memory_seq_len = memory_seq_len
        self.memory_dim = memory_dim
        
        # Load JSONL decision logs
        self.samples = []
        decision_files = list(Path(data_dir).glob("decisions_*.jsonl"))
        
        print(f"Loading data from {len(decision_files)} files...")
        for filepath in decision_files:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.samples.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract perception features
        perception = self._extract_perception(sample['perception'])
        
        # Extract or synthesize memory context
        memory = self._extract_memory(sample)
        
        # Extract action label
        action_label = self._action_to_label(sample['decision']['type'])
        
        # Extract or synthesize emotion (bootstrap from outcome)
        emotion = self._synthesize_emotion(sample)
        
        return {
            'perception': torch.tensor(perception, dtype=torch.float32),
            'memory': torch.tensor(memory, dtype=torch.float32),
            'action_label': torch.tensor(action_label, dtype=torch.long),
            'emotion': torch.tensor(emotion, dtype=torch.float32)
        }
    
    def _extract_perception(self, perception_data):
        """Convert perception JSON to vector"""
        vec = []
        
        # Position (normalized)
        pos = perception_data['position']
        vec.append(pos['x'] / 200.0)  # Assuming WORLD_WIDTH=200
        vec.append(pos['y'] / 150.0)  # Assuming WORLD_HEIGHT=150
        
        # Needs
        needs = perception_data['needs']
        vec.append(needs['hunger'])
        vec.append(needs['energy'])
        vec.append(needs['social'])
        vec.append(needs['curiosity'])
        vec.append(needs['safety'])
        
        # Time and weather
        vec.append(perception_data.get('timeOfDay', 0.5))
        vec.append(1.0 if perception_data.get('weather') == 'rain' else 0.0)
        
        # Nearby tiles (counts)
        nearby_tiles = perception_data.get('nearbyTiles', [])
        water_count = sum(1 for t in nearby_tiles if t.get('type') == 'Water')
        food_count = sum(1 for t in nearby_tiles if t.get('type') in ['BerryBush', 'Tree'])
        shelter_count = sum(1 for t in nearby_tiles if t.get('type') in ['Cave', 'Shelter'])
        
        vec.append(min(1.0, water_count / 5.0))
        vec.append(min(1.0, food_count / 5.0))
        vec.append(min(1.0, shelter_count / 3.0))
        
        # Nearby NPCs
        nearby_npcs = perception_data.get('nearbyNPCs', [])
        vec.append(min(1.0, len(nearby_npcs) / 5.0))
        
        # Emotion placeholder (bootstrapped to 0 initially)
        vec.extend([0.0, 0.0, 0.0])
        
        # Pad to fixed size
        while len(vec) < self.perception_dim:
            vec.append(0.0)
        
        return vec[:self.perception_dim]
    
    def _extract_memory(self, sample):
        """Extract or synthesize memory sequence"""
        # In Milestone 1, we don't have memory embeddings yet
        # Synthesize memory based on memory recalls
        memory = np.zeros((self.memory_seq_len, self.memory_dim), dtype=np.float32)
        
        recalls = sample['perception'].get('memoryRecalls', [])
        for i, recall in enumerate(recalls[:self.memory_seq_len]):
            # Simple encoding of memory type
            if recall == 'food':
                memory[i, 0] = 1.0
            elif recall == 'danger':
                memory[i, 1] = 1.0
            elif recall == 'npc':
                memory[i, 2] = 1.0
            elif recall == 'shelter':
                memory[i, 3] = 1.0
            
            # Add some random noise to create embedding-like structure
            memory[i, 4:] = np.random.randn(self.memory_dim - 4) * 0.1
        
        return memory
    
    def _action_to_label(self, action_type):
        """Convert action type string to label index"""
        action_map = {
            'Idle': 0,
            'Move': 1,
            'Forage': 2,
            'Eat': 3,
            'Rest': 4,
            'Explore': 5,
            'Socialize': 6,
            'BuildShelter': 7,
            'SeekShelter': 8
        }
        return action_map.get(action_type, 0)
    
    def _synthesize_emotion(self, sample):
        """Bootstrap emotion from outcome (for supervised learning)"""
        emotion = [0.0, 0.0, 0.0]  # valence, arousal, dominance
        
        # Infer emotion from needs and outcome
        needs = sample['perception']['needs']
        outcome = sample.get('outcome', {})
        
        # Valence from need satisfaction
        # Positive delta = need increased (bad) -> negative valence
        # Negative delta = need satisfied (good) -> positive valence
        need_deltas = outcome.get('needsDeltas', {})
        if need_deltas:
            avg_delta = np.mean([delta for delta in need_deltas.values()])
            emotion[0] = -np.tanh(avg_delta)  # Negate: negative delta = positive valence
        
        # Arousal from need urgency
        avg_need = (needs['hunger'] + needs['energy'] + needs['safety']) / 3.0
        emotion[1] = np.tanh(avg_need * 2 - 1)  # High needs = high arousal
        
        # Dominance from safety
        emotion[2] = np.tanh(1 - needs['safety'] * 2)  # Safe = dominant
        
        return emotion


def train_epoch(model, dataloader, optimizer, criterion_action, criterion_emotion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_emotion_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        perception = batch['perception'].to(device)
        memory = batch['memory'].to(device)
        action_label = batch['action_label'].to(device)
        emotion_target = batch['emotion'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        action_logits, emotion_pred, _ = model(perception, memory)
        
        # Compute losses
        loss_action = criterion_action(action_logits, action_label)
        loss_emotion = criterion_emotion(emotion_pred, emotion_target)
        
        # Combined loss (weighted)
        loss = loss_action + 0.5 * loss_emotion
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_action_loss += loss_action.item()
        total_emotion_loss += loss_emotion.item()
        
        _, predicted = torch.max(action_logits, 1)
        correct += (predicted == action_label).sum().item()
        total += action_label.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_action_loss = total_action_loss / len(dataloader)
    avg_emotion_loss = total_emotion_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    return avg_loss, avg_action_loss, avg_emotion_loss, accuracy


def validate(model, dataloader, criterion_action, criterion_emotion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            perception = batch['perception'].to(device)
            memory = batch['memory'].to(device)
            action_label = batch['action_label'].to(device)
            emotion_target = batch['emotion'].to(device)
            
            action_logits, emotion_pred, _ = model(perception, memory)
            
            loss_action = criterion_action(action_logits, action_label)
            loss_emotion = criterion_emotion(emotion_pred, emotion_target)
            loss = loss_action + 0.5 * loss_emotion
            
            total_loss += loss.item()
            
            _, predicted = torch.max(action_logits, 1)
            correct += (predicted == action_label).sum().item()
            total += action_label.size(0)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def export_to_onnx(model, output_path, perception_dim=20, memory_seq_len=50, memory_dim=32):
    """Export trained model to ONNX format"""
    model.eval()
    
    # Create dummy inputs
    dummy_perception = torch.randn(1, perception_dim)
    dummy_memory = torch.randn(1, memory_seq_len, memory_dim)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_perception, dummy_memory),
        output_path,
        input_names=['perception', 'memory'],
        output_names=['output'],
        dynamic_axes={
            'perception': {0: 'batch'},
            'memory': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=12
    )
    
    print(f"Model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train NPC Brain')
    parser.add_argument('--data-dir', type=str, default='data_logs',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = NPCDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("Error: No training data found!")
        print(f"Please run the simulator to generate data in {args.data_dir}/")
        return
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Create model
    print("Creating model...")
    model = create_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss functions and optimizer
    criterion_action = nn.CrossEntropyLoss()
    criterion_emotion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_action_loss, train_emotion_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_action, criterion_emotion, device
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion_action, criterion_emotion, device
        )
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (Action: {train_action_loss:.4f}, "
              f"Emotion: {train_emotion_loss:.4f}), Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, 'npc_brain_best.pth'))
            print(f"  -> Saved best model")
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(args.output_dir, 'npc_brain_final.pth'))
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    onnx_path = os.path.join(args.output_dir, 'npc_brain.onnx')
    export_to_onnx(model, onnx_path)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
