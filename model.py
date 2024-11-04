import torch
import torch.nn as nn
import math
import torch.nn.functional as F



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
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sin to all even positions
        pos_enc[:, 0::2] = torch.sin(position * denominator)
        pos_enc[:, 1::2] = torch.cos(position * denominator)

        # add a batch dimension: new shape is (1, sequence_length, d_model)
        pos_enc = pos_enc.unsqueeze(0)

        # this saves the positional encoding tensor in memory
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        # pos_enc should match the shape (batch_size, sequence_length, d_model)
        # Add requires_grad_(False) to prevent gradients from being calculated
        x = x + self.pos_enc[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10 ** (-6)):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        standard_deviation = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (standard_deviation + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, ffn_dim)  # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ffn_dim, d_model)  # W2 and b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear layers for query, key, value
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)

        # Output linear layer
        self.out_layer = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query_layer(query)  # (batch_size, query_len, embedding_dim)
        key = self.key_layer(key)        # (batch_size, key_len, embedding_dim)
        value = self.value_layer(value)  # (batch_size, key_len, embedding_dim)
        
        # Split into heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        # query shape: (batch_size, num_heads, query_len, head_dim)
        # key shape: (batch_size, num_heads, key_len, head_dim)
        # einsum result shape: (batch_size, num_heads, query_len, key_len)
        scores = torch.einsum("bhqd, bhkd -> bhqk", query, key) / (self.head_dim ** 0.5)

        # Apply mask (if any)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute attention output
        # attention_weights shape: (batch_size, num_heads, query_len, key_len)
        # value shape: (batch_size, num_heads, key_len, head_dim)
        # einsum result shape: (batch_size, num_heads, query_len, head_dim)
        attention_output = torch.einsum("bhqk, bhkd -> bhqd", attention_weights, value)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final linear layer
        output = self.out_layer(attention_output)
        
        return output, attention_weights

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))[0])


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, source_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, source_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, source_mask
            ),
        )
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocabulary_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocabulary_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embedding: InputEmbedding,
        target_embedding: InputEmbedding,
        source_positional_embedding: PositionalEncoding,
        target_positional_embedding: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_embedding = target_embedding
        self.source_embedding = source_embedding
        self.target_positional_embedding = target_positional_embedding
        self.source_positional_embedding = source_positional_embedding
        self.projection_layer = projection_layer

    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_positional_embedding(source)
        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_positional_embedding(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)


def transformerBuilder(
    source_vocabulary_size: int,
    target_vocabulary_size: int,
    source_sequence_length: int,
    target_sequence_length: int,
    d_model: int = 512,
    num_layers: int = 6,
    heads: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    source_embedding = InputEmbedding(d_model, source_vocabulary_size)
    target_embedding = InputEmbedding(d_model, target_vocabulary_size)

    # create positional encodings
    source_positional_encoding = PositionalEncoding(
        d_model, source_sequence_length, dropout
    )
    target_positional_encoding = PositionalEncoding(
        d_model, target_sequence_length, dropout
    )

    # create encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layyer
    projection_layer = ProjectionLayer(d_model, target_vocabulary_size)

    # create the transformer
    transformer = Transformer(
        encoder,
        decoder,
        source_embedding,
        target_embedding,
        source_positional_encoding,
        target_positional_encoding,
        projection_layer,
    )
    # Initialize parameters
    for x in transformer.parameters():
        if x.dim() > 1:
            nn.init.xavier_uniform_(x)

    return transformer
