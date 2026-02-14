"""
Fine-tune NPC Brain - Create unique personalities from individual experience logs
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_architecture import create_model
from train_npc_brain import NPCDataset, train_epoch, validate, export_to_onnx


class PersonalizedNPCDataset(NPCDataset):
    """Dataset filtered for specific NPC ID"""
    
    def __init__(self, data_dir, npc_id, perception_dim=20, memory_seq_len=50, memory_dim=32):
        super().__init__(data_dir, perception_dim, memory_seq_len, memory_dim)
        
        # Filter samples for specific NPC
        self.samples = [s for s in self.samples if s.get('npcId') == npc_id]
        print(f"Filtered to {len(self.samples)} samples for NPC {npc_id}")


def fine_tune(base_model_path, npc_id, data_dir, output_dir, epochs=20, lr=0.0001, device='cpu'):
    """Fine-tune base model on individual NPC's experience"""
    
    # Load base model
    print(f"Loading base model from {base_model_path}...")
    model = create_model()
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model = model.to(device)
    
    # Load personalized dataset
    print(f"Loading data for NPC {npc_id}...")
    dataset = PersonalizedNPCDataset(data_dir, npc_id)
    
    if len(dataset) < 10:
        print(f"Warning: Only {len(dataset)} samples for NPC {npc_id}, "
              f"fine-tuning may not be effective")
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_action = nn.CrossEntropyLoss()
    criterion_emotion = nn.MSELoss()
    
    # Fine-tuning loop
    print(f"\nFine-tuning for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_action_loss, train_emotion_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_action, criterion_emotion, device
        )
        
        print(f"Epoch {epoch+1}/{epochs}: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        if train_loss < best_loss:
            best_loss = train_loss
            # Save personalized model
            output_path = os.path.join(output_dir, f'npc_brain_{npc_id}.pth')
            torch.save(model.state_dict(), output_path)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, f'npc_brain_{npc_id}.onnx')
    export_to_onnx(model, onnx_path)
    
    print(f"\nFine-tuning complete for NPC {npc_id}!")
    print(f"Personalized model saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune NPC Brain for specific personality')
    parser.add_argument('--base-model', type=str, required=True,
                        help='Path to base trained model (.pth file)')
    parser.add_argument('--npc-id', type=int, required=True,
                        help='NPC ID to fine-tune for')
    parser.add_argument('--data-dir', type=str, default='data_logs',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (lower than base training)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fine-tune
    fine_tune(
        args.base_model,
        args.npc_id,
        args.data_dir,
        args.output_dir,
        args.epochs,
        args.lr,
        device
    )


if __name__ == '__main__':
    main()
