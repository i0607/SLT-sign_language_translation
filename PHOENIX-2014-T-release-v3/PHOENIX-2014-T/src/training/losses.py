import torch
import torch.nn as nn
from torch.nn import CTCLoss, CrossEntropyLoss

class JointLoss(nn.Module):
    def __init__(self, lambda_r=5.0, lambda_t=1.0, blank_idx=0, pad_idx=-100):
        """
        Joint loss combining CTC for recognition and CrossEntropy for translation
        
        Args:
            lambda_r: Weight for recognition loss (paper uses 5.0)
            lambda_t: Weight for translation loss (paper uses 1.0)
            blank_idx: CTC blank token index
            pad_idx: Padding index to ignore in translation loss
        """
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_t = lambda_t
        
        # Recognition loss (CTC)
        self.ctc_loss = CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        
        # Translation loss (CrossEntropy)
        self.ce_loss = CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')
    
    def forward(self, ctc_log_probs, translation_logits, 
                target_glosses, target_words, 
                input_lengths, target_gloss_lengths):
        """
        Calculate joint loss
        
        Args:
            ctc_log_probs: (batch, time, vocab_gloss) - CTC predictions
            translation_logits: (batch, seq_len, vocab_word) - Translation predictions
            target_glosses: (batch, max_gloss_len) - Target gloss IDs
            target_words: (batch, max_word_len) - Target word IDs
            input_lengths: (batch,) - Actual frame counts
            target_gloss_lengths: (batch,) - Actual gloss sequence lengths
        
        Returns:
            total_loss, recognition_loss, translation_loss
        """
        
        # 1. Recognition Loss (CTC)
        ctc_log_probs_t = ctc_log_probs.transpose(0, 1)  # (time, batch, vocab)
        
        recognition_loss = self.ctc_loss(
            ctc_log_probs_t,
            target_glosses,
            input_lengths,
            target_gloss_lengths
        )
        
        # 2. Translation Loss (CrossEntropy)
        target_words_shifted = target_words[:, 1:]  # Remove <SOS>
        
        batch_size, seq_len, vocab_size = translation_logits.shape
        translation_logits_flat = translation_logits.reshape(-1, vocab_size)
        target_words_flat = target_words_shifted.reshape(-1)
        
        translation_loss = self.ce_loss(translation_logits_flat, target_words_flat)
        
        # 3. Combined Loss
        total_loss = self.lambda_r * recognition_loss + self.lambda_t * translation_loss
        
        return total_loss, recognition_loss, translation_loss