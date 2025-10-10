import torch
import torch.nn as nn
import math
from src.models.positional_encoding import PositionalEncoding

class SLTT(nn.Module):
    def __init__(self, vocab_size=2897, hidden_dim=512, num_layers=3, num_heads=8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # Add padding_idx to handle -100 values
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, tgt_tokens, encoder_output, tgt_mask=None):
        # Clamp tokens to valid range [0, vocab_size)
        tgt_tokens = torch.clamp(tgt_tokens, min=0, max=self.word_embedding.num_embeddings-1)
        
        # Embed target words
        tgt_emb = self.word_embedding(tgt_tokens) * math.sqrt(self.hidden_dim)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Decode
        decoder_output = self.transformer_decoder(
            tgt_emb, encoder_output, tgt_mask=tgt_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        return output