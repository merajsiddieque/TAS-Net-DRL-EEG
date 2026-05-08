import torch
import torch.nn as nn

__all__ = ['DSN']


class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc_actor = nn.Linear(hid_dim*2, 2)
        self.fc_critic = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        logits = self.fc_actor(h)
        v = self.fc_critic(h)
        return logits, v
