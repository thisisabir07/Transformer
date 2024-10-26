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


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self. dropout = nn.Dropout(dropout)
        assert d_model % heads == 0, "D_model should be divisible by the number of heads"

        dim_q = self.d_model // self.heads
        self.weight_q = nn.Parameter(torch.randn(size=(heads, d_model, dim_q),generator=torch.random.manual_seed(0)))
        self.weight_k = nn.Parameter(torch.randn(size=(heads, d_model, dim_q),generator=torch.random.manual_seed(1)))
        self.weight_v = nn.Parameter(torch.randn(size=(heads, d_model, dim_q),generator=torch.random.manual_seed(2)))
        self.weight_out = nn.Parameter(torch.randn(size=(d_model,d_model), generator=torch.random.manual_seed(3)))

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        
        dim_k = query.shape[-1]

        attention_scores = (torch.einsum('bhsq, bhtq -> bhst', query, key)) / torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # dimention of attention_score is: batch_size (b), heads (h), sequence_length (s), sequence_length (s)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Dimensions:
        # attention_scores: batch_size (b), heads (h), sequence_length (s), sequence_length (s)
        # value: batch_size (b), heads (h), sequence_length (s), dim_q (q)
        # output: batch_size (b), heads (h), sequence_length (s), dim_q (q)

        output = torch.einsum('bhss,bhsq->bhsq', attention_scores, value)
        return output, attention_scores



    def forward(self, q, k ,v, mask):

        # q, k, v = batch_size (b), sequence_length (s), d_model (d)
        # weight_q, weight_k, weight_v = heads (h), dim_q (q), d_model (d)
        # Q, K, V = batch_size (b), heads (h), sequence_length (s), dim_q (q)

        # From the above dimensions it is clear that we are trying to multiply the matrices accross the d_model dimension and preserve all other dimensions.

        Q = torch.einsum('bsd, hdq -> bhsq', q, self.weight_q)
        K = torch.einsum('bsd, hdq -> bhsq', k, self.weight_k)
        V = torch.einsum('bsd, hdq -> bhsq', v, self.weight_v)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(Q, K, V, mask, self.dropout)







