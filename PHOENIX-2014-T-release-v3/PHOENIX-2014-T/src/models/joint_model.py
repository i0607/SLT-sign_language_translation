import torch
import torch.nn as nn
from src.models.slrt import SLRT
from src.models.sltt import SLTT

class SignLanguageTransformer(nn.Module):
    def __init__(self, gloss_vocab_size, word_vocab_size, 
                 input_dim=1024, hidden_dim=512, num_layers=3, num_heads=8):
        super().__init__()
        self.slrt = SLRT(input_dim, hidden_dim, num_layers, num_heads, gloss_vocab_size)
        self.sltt = SLTT(word_vocab_size, hidden_dim, num_layers, num_heads)
        
    def forward(self, frames, target_words=None):
        # Get encoder output
        encoder_output = self.slrt.encode(frames)
        
        # Recognition: CTC predictions
        ctc_logits = self.slrt.ctc_classifier(encoder_output)
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
        
        if target_words is not None:
            # Translation: decoder predictions (training mode)
            tgt_input = target_words[:, :-1]  # Remove last token (keep <SOS>, remove last word)
            
            # Replace -100 padding with 0 for embedding layer
            tgt_input = torch.where(tgt_input == -100, torch.zeros_like(tgt_input), tgt_input)
            
            seq_len = tgt_input.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=frames.device) * float('-inf'), diagonal=1)
            
            translation_logits = self.sltt(tgt_input, encoder_output, tgt_mask)
            return ctc_log_probs, translation_logits
        else:
            return ctc_log_probs