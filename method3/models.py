import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticRNN(nn.Module):
    """
    Base Actor-Critic sequence model leveraging RNN core. 
    Outputs both selecting logits (Actor) and state value (Critic).
    """
    def __init__(self, in_dim=192, hid_dim=256, num_layers=1, cell='gru'):
        super(ActorCriticRNN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        
        # Dual heads for Actor-Critic
        self.fc_actor = nn.Linear(hid_dim * 2, 1)
        self.fc_critic = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        # Actor probabilities (bounded between 0 and 1)
        probs = torch.sigmoid(self.fc_actor(h))
        # Critic values (unbounded state-value estimate V(s))
        values = self.fc_critic(h)
        return probs, values

class HighLevelActorCritic(ActorCriticRNN):
    """
    High-level policy network. Responsible for deciding whether to select a coarse 
    segment chunk of the sequence.
    """
    pass

class LowLevelActorCritic(ActorCriticRNN):
    """
    Low-level policy network. Conditioned on an active high-level segment, 
    evaluates frame-level selections inside the segment.
    """
    pass

import math

class HierarchicalNetworkProxy(nn.Module):
    def __init__(self, high, low, seg_len=16):
        super().__init__()
        self.high = high
        self.low = low
        self.seg_len = seg_len

    def forward(self, seq_feats):
        # Emulate original probability sequences for external evaluate tools
        # seq_feats shape: (1, seq_len, dim)
        seq_f = seq_feats.squeeze(0)
        num_segments = math.ceil(seq_f.shape[0] / self.seg_len)
        all_probs = []
        for s_i in range(num_segments):
            start = s_i * self.seg_len
            end = min(start + self.seg_len, seq_f.shape[0])
            seg_frames = seq_f[start:end]
            
            seg_feat = seg_frames.mean(dim=0, keepdim=True).unsqueeze(0)
            h_prob, _ = self.high(seg_feat)
            
            l_probs, _ = self.low(seg_frames.unsqueeze(0))
            # Chain Rule: P(frame | selected segment) * P(segment)
            adjusted_probs = l_probs.squeeze(0) * h_prob.squeeze().item()
            all_probs.append(adjusted_probs)
            
        return torch.cat(all_probs, dim=0).unsqueeze(0)

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        # p = F.sigmoid(self.fc(h))
        p = torch.sigmoid(self.fc(h))
        # pdb.set_trace()
        return p