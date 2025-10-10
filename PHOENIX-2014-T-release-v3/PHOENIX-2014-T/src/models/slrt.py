import torch
import torch.nn as nn
from src.models.positional_encoding import PositionalEncoding

class SLRT(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=3, 
                 num_heads=8, gloss_vocab_size=1811):
        """
        Sign Language Recognition Transformer
        
        Args:
            input_dim: CNN output dimension
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            gloss_vocab_size: Size of gloss vocabulary
        """
        super().__init__()
        
        # CNN to extract features from video frames
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, input_dim)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # CTC output layer
        self.ctc_classifier = nn.Linear(hidden_dim, gloss_vocab_size + 1)  # +1 for blank
        
    def encode(self, frames):
        """
        Encode video frames to hidden representations
        
        Args:
            frames: (batch, time, height, width, channels)
        Returns:
            transformer_out: (batch, time, hidden_dim)
        """
        batch_size, seq_len, h, w, c = frames.shape
        frames = frames.view(-1, c, h, w)
        cnn_features = self.cnn(frames)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        x = self.input_projection(cnn_features)
        x = self.pos_encoding(x)
        transformer_out = self.transformer(x)
        return transformer_out

    def forward(self, frames):
        """
        Forward pass for recognition
        
        Args:
            frames: (batch, time, height, width, channels)
        Returns:
            ctc_log_probs: (batch, time, vocab_size+1)
        """
        transformer_out = self.encode(frames)
        ctc_logits = self.ctc_classifier(transformer_out)
        return torch.log_softmax(ctc_logits, dim=-1)