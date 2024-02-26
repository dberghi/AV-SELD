import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class CMAF_Layer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(CMAF_Layer, self).__init__()
        self.MHSA = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) # self-attention
        self.MHCA = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) # cross-attention

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = F.relu

    def forward(self, alpha, beta): # forward two generic modalities alpha and beta
        #alpha = alpha.transpose(0, 1)  # B, T, C -> T, B, C needed if batch_first=False
        #beta = beta.transpose(0, 1)  # B, T, C -> T, B, C needed if batch_first=False
        alpha = self.pos_encoder(alpha)
        beta = self.pos_encoder(beta)
        x_sa, _ = self.MHSA(alpha, alpha, alpha, attn_mask=None, key_padding_mask=None)
        x_ca, _ = self.MHCA(alpha, beta, beta, attn_mask=None, key_padding_mask=None)
        x = self.norm1(alpha + x_sa + x_ca)

        x_mlp = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = x + self.dropout2(x_mlp)
        x = self.norm2(x)

        #x = x.transpose(0, 1)  # T, B, C -> B, T, C needed if batch_first=False
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0, 1)  # B, T, C -> T, B, C
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.transpose(0, 1)  # T, B, C -> B, T, C
        return x