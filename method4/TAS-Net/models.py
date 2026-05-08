import torch
import torch.nn as nn
import math

__all__ = ['DSN']

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class DSN(nn.Module):
    """Deep Summarization Network (Transformer + Actor-Critic)"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=2, nhead=8, cell='transformer'):
        super(DSN, self).__init__()
        
        self.pos_encoder = PositionalEncoding(in_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nhead, dim_feedforward=hid_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Actor head for probabilities
        self.actor_fc = nn.Linear(in_dim, 1)
        
        # Critic head for state values
        self.critic_fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim)
        x = self.pos_encoder(x)
        h = self.transformer_encoder(x)
        
        p = torch.sigmoid(self.actor_fc(h))
        v = self.critic_fc(h)
        return p, v, h
