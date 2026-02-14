"""
Neural Network Architecture for NPC Brain
Transformer-based decision network with memory attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class MemoryAttentionBlock(nn.Module):
    """Attention mechanism over episodic memories"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, query, memory, mask=None):
        # Self-attention with memory as key/value
        attn_out, attn_weights = self.attention(query, memory, memory, 
                                                 key_padding_mask=mask,
                                                 need_weights=True)
        query = self.norm1(query + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(query)
        output = self.norm2(query + ffn_out)
        
        return output, attn_weights


class NPCBrainModel(nn.Module):
    """
    Transformer-based NPC decision model
    
    Input:
        - perception: current perception vector (20 dimensions)
        - memory: sequence of memory embeddings (50 x 32 dimensions)
    
    Output:
        - action_probs: probability distribution over 9 actions
        - emotion: 3D emotional state (valence, arousal, dominance)
    """
    
    def __init__(self, perception_dim=20, memory_seq_len=50, memory_dim=32,
                 d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.perception_dim = perception_dim
        self.memory_seq_len = memory_seq_len
        self.memory_dim = memory_dim
        self.d_model = d_model
        
        # Perception encoder
        self.perception_encoder = nn.Sequential(
            nn.Linear(perception_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Memory encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for memory sequence
        self.pos_encoding = PositionalEncoding(d_model, max_len=memory_seq_len)
        
        # Transformer blocks with memory attention
        self.attention_blocks = nn.ModuleList([
            MemoryAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 9)  # 9 action types
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),  # valence, arousal, dominance
            nn.Tanh()  # Emotions in [-1, 1]
        )
    
    def forward(self, perception, memory, memory_mask=None):
        """
        Args:
            perception: (batch, perception_dim)
            memory: (batch, memory_seq_len, memory_dim)
            memory_mask: (batch, memory_seq_len) - True for padding positions
        
        Returns:
            action_logits: (batch, 9)
            emotion: (batch, 3)
            attention_weights: list of (batch, memory_seq_len) from each layer
        """
        # Encode perception as query
        query = self.perception_encoder(perception)  # (batch, d_model)
        query = query.unsqueeze(1)  # (batch, 1, d_model)
        
        # Encode memory sequence
        batch_size = memory.size(0)
        memory_encoded = self.memory_encoder(memory)  # (batch, seq_len, d_model)
        memory_encoded = self.pos_encoding(memory_encoded)
        
        # Apply attention blocks
        attention_weights_list = []
        for block in self.attention_blocks:
            query, attn_weights = block(query, memory_encoded, mask=memory_mask)
            attention_weights_list.append(attn_weights.squeeze(1))  # (batch, seq_len)
        
        # Extract final representation
        context = query.squeeze(1)  # (batch, d_model)
        
        # Generate outputs
        action_logits = self.action_head(context)  # (batch, 9)
        emotion = self.emotion_head(context)  # (batch, 3)
        
        return action_logits, emotion, attention_weights_list


def create_model(perception_dim=20, memory_seq_len=50, memory_dim=32,
                 d_model=128, n_heads=4, n_layers=2, dropout=0.1):
    """Factory function to create model with default parameters"""
    return NPCBrainModel(
        perception_dim=perception_dim,
        memory_seq_len=memory_seq_len,
        memory_dim=memory_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model()
    
    # Create dummy inputs
    batch_size = 4
    perception = torch.randn(batch_size, 20)
    memory = torch.randn(batch_size, 50, 32)
    
    # Forward pass
    action_logits, emotion, attention_weights = model(perception, memory)
    
    print(f"Model created successfully!")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Emotion shape: {emotion.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test softmax for action probabilities
    action_probs = F.softmax(action_logits, dim=-1)
    print(f"\nSample action probabilities:\n{action_probs[0]}")
    print(f"\nSample emotion:\n{emotion[0]}")
