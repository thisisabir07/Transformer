import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocabulary_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, d_model)


    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape: (sequence_length, d_model)
        pos_enc = torch.zeros(sequence_length, d_model)

        # create a vector of shape(sequence_length, 1)
        position = torch.arange(0, sequence_length, dtype = torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        # apply sin to all even positons
        pos_enc[:, 0::2] = torch.sin(position * denominator)
        pos_enc[:, 1::2] = torch.cos(position * denominator)

        pos_enc.unsqueeze(0) # the new dimension of pos_enc is (1, sequence_length, d_model)

        # this saves the positional encoding tensor in memory
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, epsilon: float = 10**(-6)):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        standard_deviation = x.std(dim = -1, keepdim = True)

        return self.alpha * (x-mean) / (standard_deviation + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, ffn_dim) #W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ffn_dim, d_model) #W2 and b2


    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

