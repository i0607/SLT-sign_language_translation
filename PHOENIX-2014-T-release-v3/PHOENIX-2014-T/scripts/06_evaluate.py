"""
Main Evaluation Script for Sign Language Translation Model
Evaluates trained model on test/dev sets with comprehensive metrics

Usage:
    python scripts/06_evaluate.py --split test
    python scripts/06_evaluate.py --split dev --checkpoint path/to/model.pth
"""

import sys
import os

# Fix Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
import torch
import pandas as pd
import numpy as np
import json
import glob
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import ProperSignDataset
from src.data.vocabulary import build_vocabularies
from src.models.joint_model import SignLanguageTransformer
from src.training.losses import JointLoss


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualization"""
    
    def __init__(self, checkpoint_path, config_path):
        """
        Initialize evaluator
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to config.yaml
        """
        self.checkpoint_path = checkpoint_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print("="*80)
        print("Model Evaluation Setup")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}\n")
        
        # Load vocabularies from training data
        train_data = pd.read_csv(self.config['data']['train_annotations'])
        self.gloss_vocab, self.word_vocab = build_vocabularies(train_data)
        
        # Create reverse vocabularies for decoding
        self.gloss_id_to_token = {v: k for k, v in self.gloss_vocab.items()}
        self.word_id_to_token = {v: k for k, v in self.word_vocab.items()}
        
        print(f"Gloss vocabulary size: {len(self.gloss_vocab)}")
        print(f"Word vocabulary size: {len(self.word_vocab)}\n")
        
        # Load model
        self.model = self._load_model()
        
        # Loss function
        self.criterion = JointLoss(
            lambda_r=self.config['training']['lambda_recognition'],
            lambda_t=self.config['training']['lambda_translation']
        )
        
        # Output directory (in same experiment folder)
        exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        self.output_dir = os.path.join(exp_dir, 'evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}\n")
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        model = SignLanguageTransformer(
            gloss_vocab_size=len(self.gloss_vocab),
            word_vocab_size=len(self.word_vocab),
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads']
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        return model
    
    def collate_fn(self, batch):
        """Collate function for DataLoader - same as training"""
        max_frames = max(item['frames'].shape[0] for item in batch)
        max_glosses = max(len(item['gloss_ids']) for item in batch)
        max_words = max(len(item['word_ids']) for item in batch)
        
        batch_size = len(batch)
        _, height, width, channels = batch[0]['frames'].shape
        
        frames_batch = torch.zeros(batch_size, max_frames, height, width, channels, dtype=torch.float32)
        glosses_batch = torch.zeros(batch_size, max_glosses, dtype=torch.long)
        words_batch = torch.full((batch_size, max_words), -100, dtype=torch.long)
        
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        gloss_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, item in enumerate(batch):
            num_frames = item['frames'].shape[0]
            frames_batch[i, :num_frames] = item['frames']
            input_lengths[i] = num_frames
            
            num_glosses = len(item['gloss_ids'])
            glosses_batch[i, :num_glosses] = item['gloss_ids']
            gloss_lengths[i] = num_glosses
            
            num_words = len(item['word_ids'])
            words_batch[i, :num_words] = item['word_ids']
        
        return {
            'frames': frames_batch,
            'glosses': glosses_batch,
            'words': words_batch,
            'input_lengths': input_lengths,
            'gloss_lengths': gloss_lengths
        }
    
    def decode_ctc(self, ctc_log_probs, input_lengths):
        """Decode CTC predictions to gloss sequences (greedy decoding)"""
        batch_size = ctc_log_probs.size(0)
        predictions = []
        
        # DEBUG: Check the probabilities for first sample
        if batch_size > 0:
            length = input_lengths[0].item()
            sample_probs = ctc_log_probs[0, :length, :]
            
            print(f"\n{'='*80}")
            print("DEBUG: CTC Decoding Analysis (First Sample)")
            print('='*80)
            print(f"CTC output shape: {ctc_log_probs.shape}")
            print(f"Input length: {length}")
            print(f"Vocab size: {sample_probs.shape[-1]}")
            
            # Check top predictions at each timestep
            pred_ids = torch.argmax(sample_probs, dim=-1).cpu().numpy()
            print(f"\nRaw predictions (first 30 frames): {pred_ids[:30]}")
            
            # Check distribution of predictions
            unique, counts = np.unique(pred_ids, return_counts=True)
            print(f"\nToken distribution across all frames:")
            for token_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
                token_name = self.gloss_id_to_token.get(int(token_id), '<UNK>')
                percentage = (count / len(pred_ids)) * 100
                print(f"  Token {token_id:3d} ({token_name:20s}): {count:4d} times ({percentage:5.1f}%)")
            
            # Check max probabilities
            max_probs = torch.max(torch.exp(sample_probs), dim=-1)[0].cpu().numpy()
            print(f"\nConfidence scores:")
            print(f"  Mean max prob: {np.mean(max_probs):.4f}")
            print(f"  Min max prob:  {np.min(max_probs):.4f}")
            print(f"  Max max prob:  {np.max(max_probs):.4f}")
            
            # Check if blank is dominating
            blank_prob = torch.exp(sample_probs[:, 0]).cpu().numpy()
            print(f"\nBlank token (ID 0) probabilities:")
            print(f"  Mean: {np.mean(blank_prob):.4f}")
            print(f"  Frames where blank is max: {np.sum(pred_ids == 0)} / {length}")
        
        for i in range(batch_size):
            length = input_lengths[i].item()
            sample_probs = ctc_log_probs[i, :length, :]
            pred_ids = torch.argmax(sample_probs, dim=-1).cpu().numpy()
            
            # Remove blanks and consecutive duplicates
            decoded = []
            prev_id = None
            for pred_id in pred_ids:
                if pred_id != 0 and pred_id != prev_id:  # 0 is blank
                    token = self.gloss_id_to_token.get(int(pred_id), '<UNK>')
                    decoded.append(token)
                prev_id = pred_id
            
            if i == 0:
                print(f"\nAfter CTC decoding:")
                print(f"  Raw sequence length: {len(pred_ids)}")
                print(f"  Non-blank predictions: {np.sum(pred_ids != 0)}")
                print(f"  After collapse: {len(decoded)}")
                print(f"  Decoded glosses: {decoded}")
                print('='*80 + '\n')
            
            predictions.append(decoded)
        
        return predictions
        
    def decode_translation(self, logits):
        """Decode translation logits to word sequences"""
        batch_size = logits.size(0)
        predictions = []
        
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        
        for i in range(batch_size):
            decoded = []
            for token_id in pred_ids[i]:
                if token_id == self.word_vocab['<EOS>']:
                    break
                if token_id not in [self.word_vocab['<SOS>'], -100]:
                    word = self.word_id_to_token.get(token_id, '<UNK>')
                    decoded.append(word)
            predictions.append(decoded)
        
        return predictions
    
    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate using dynamic programming"""
        r = len(reference)
        h = len(hypothesis)
        
        d = np.zeros((r + 1, h + 1))
        
        for i in range(r + 1):
            d[i][0] = i
        for j in range(h + 1):
            d[0][j] = j
        
        for i in range(1, r + 1):
            for j in range(1, h + 1):
                if reference[i-1] == hypothesis[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        
        return d[r][h] / r if r > 0 else 0
    
    def calculate_bleu(self, reference, hypothesis, max_n=4):
        """Calculate BLEU score (simplified)"""
        if len(hypothesis) == 0:
            return 0.0
        
        ref_len = len(reference)
        hyp_len = len(hypothesis)
        
        # Brevity penalty
        if hyp_len > ref_len:
            bp = 1
        else:
            bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0
        
        # N-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(reference, n)
            hyp_ngrams = self._get_ngrams(hypothesis, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0)
                continue
            
            matches = sum(min(ref_ngrams[ng], hyp_ngrams[ng]) for ng in hyp_ngrams)
            precision = matches / sum(hyp_ngrams.values())
            precisions.append(precision)
        
        if min(precisions) == 0:
            return 0.0
        
        log_precisions = [np.log(p) for p in precisions if p > 0]
        geo_mean = np.exp(sum(log_precisions) / len(log_precisions))
        
        return bp * geo_mean
    
    def _get_ngrams(self, tokens, n):
        """Get n-gram counts"""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def evaluate_dataset(self, split='test'):
        """Evaluate model on dataset split"""
        print(f"\n{'='*80}")
        print(f"Evaluating on {split.upper()} set")
        print('='*80)
        
        # Create dataset
        if split == 'test':
            annotations = self.config['data']['test_annotations']
            features = self.config['data']['test_features']
        else:
            annotations = self.config['data']['dev_annotations']
            features = self.config['data']['dev_features']
        
        dataset = ProperSignDataset(
            annotations,
            features,
            self.gloss_vocab,
            self.word_vocab
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn
        )
        
        print(f"Dataset size: {len(dataset)} samples\n")
        
        # Metrics storage
        total_loss = 0
        total_rec_loss = 0
        total_trans_loss = 0
        num_batches = 0
        
        all_gloss_wer = []
        all_translation_wer = []
        all_translation_bleu = []
        all_predictions = []
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating")):
                try:
                    frames = batch['frames'].to(self.device)
                    glosses = batch['glosses'].to(self.device)
                    words = batch['words'].to(self.device)
                    input_lengths = batch['input_lengths'].to(self.device)
                    gloss_lengths = batch['gloss_lengths'].to(self.device)
                    
                    # Forward pass
                    ctc_log_probs, translation_logits = self.model(frames, words)
                    
                    # Calculate loss
                    loss, rec_loss, trans_loss = self.criterion(
                        ctc_log_probs,
                        translation_logits,
                        glosses,
                        words,
                        input_lengths,
                        gloss_lengths
                    )
                    
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        total_rec_loss += rec_loss.item()
                        total_trans_loss += trans_loss.item()
                        num_batches += 1
                    
                    # Decode predictions
                    pred_glosses = self.decode_ctc(ctc_log_probs, input_lengths)
                    pred_translations = self.decode_translation(translation_logits)
                    
                    # Get ground truth and calculate metrics
                    for i in range(len(pred_glosses)):
                        gt_gloss_ids = glosses[i].cpu().numpy()
                        gt_gloss_len = gloss_lengths[i].item()
                        gt_glosses = [self.gloss_id_to_token.get(int(g), '<UNK>') 
                                     for g in gt_gloss_ids[:gt_gloss_len] if g != 0]
                        
                        gt_word_ids = words[i].cpu().numpy()
                        gt_words = [self.word_id_to_token.get(int(w), '<UNK>') 
                                   for w in gt_word_ids if w not in [-100, 
                                   self.word_vocab['<SOS>'], 
                                   self.word_vocab['<EOS>']]]
                        
                        # Calculate metrics
                        gloss_wer = self.calculate_wer(gt_glosses, pred_glosses[i])
                        trans_wer = self.calculate_wer(gt_words, pred_translations[i])
                        trans_bleu = self.calculate_bleu(gt_words, pred_translations[i])
                        
                        all_gloss_wer.append(gloss_wer)
                        all_translation_wer.append(trans_wer)
                        all_translation_bleu.append(trans_bleu)
                        
                        all_predictions.append({
                            'sample_idx': batch_idx * len(pred_glosses) + i,
                            'gt_glosses': ' '.join(gt_glosses),
                            'pred_glosses': ' '.join(pred_glosses[i]),
                            'gt_translation': ' '.join(gt_words),
                            'pred_translation': ' '.join(pred_translations[i]),
                            'gloss_wer': gloss_wer,
                            'translation_wer': trans_wer,
                            'translation_bleu': trans_bleu
                        })
                        
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Calculate average metrics
        results = {
            'split': split,
            'num_samples': len(dataset),
            'avg_loss': total_loss / num_batches if num_batches > 0 else float('inf'),
            'avg_recognition_loss': total_rec_loss / num_batches if num_batches > 0 else float('inf'),
            'avg_translation_loss': total_trans_loss / num_batches if num_batches > 0 else float('inf'),
            'gloss_wer': np.mean(all_gloss_wer) * 100,
            'translation_wer': np.mean(all_translation_wer) * 100,
            'translation_bleu': np.mean(all_translation_bleu) * 100,
            'predictions': all_predictions
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*80)
        print(f"Evaluation Results - {results['split'].upper()} Set")
        print("="*80)
        print(f"Number of samples: {results['num_samples']}")
        print(f"\nLoss Metrics:")
        print(f"  Combined Loss:     {results['avg_loss']:.4f}")
        print(f"  Recognition Loss:  {results['avg_recognition_loss']:.4f}")
        print(f"  Translation Loss:  {results['avg_translation_loss']:.4f}")
        print(f"\nRecognition Metrics (Glosses):")
        print(f"  Word Error Rate:   {results['gloss_wer']:.2f}%")
        print(f"\nTranslation Metrics:")
        print(f"  Word Error Rate:   {results['translation_wer']:.2f}%")
        print(f"  BLEU Score:        {results['translation_bleu']:.2f}%")
        print("="*80)
    
    def save_results(self, results):
        """Save results to files"""
        # Save metrics (without predictions list)
        results_to_save = {k: v for k, v in results.items() if k != 'predictions'}
        
        output_path = os.path.join(self.output_dir, f'{results["split"]}_results.json')
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame(results['predictions'])
        print(predictions_df.head())
        csv_path = os.path.join(self.output_dir, f'{results["split"]}_predictions.csv')
        predictions_df.to_csv(csv_path, index=False)
        print(f"✓ Predictions saved to: {csv_path}")
    
    def visualize_results(self, results):
        """Create visualization plots"""
        print(f"\nGenerating visualizations...")
        
        df = pd.DataFrame(results['predictions'])
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Evaluation Results - {results["split"].upper()} Set', 
                    fontsize=16, fontweight='bold')
        
        # Gloss WER Distribution
        axes[0, 0].hist(df['gloss_wer'] * 100, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(results['gloss_wer'], color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {results["gloss_wer"]:.2f}%')
        axes[0, 0].set_xlabel('Gloss WER (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Gloss Recognition - WER Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Translation WER Distribution
        axes[0, 1].hist(df['translation_wer'] * 100, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(results['translation_wer'], color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {results["translation_wer"]:.2f}%')
        axes[0, 1].set_xlabel('Translation WER (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Translation - WER Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # BLEU Score Distribution
        axes[1, 0].hist(df['translation_bleu'] * 100, bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(results['translation_bleu'], color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {results["translation_bleu"]:.2f}%')
        axes[1, 0].set_xlabel('BLEU Score (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Translation - BLEU Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics Summary
        metrics = ['Gloss WER', 'Trans WER', 'BLEU']
        values = [results['gloss_wer'], results['translation_wer'], results['translation_bleu']]
        colors = ['blue', 'green', 'purple']
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].set_title('Metrics Summary')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}%',
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'{results["split"]}_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {plot_path}")
    
    def show_sample_predictions(self, results, num_samples=5):
        """Show sample predictions"""
        print(f"\n{'='*80}")
        print(f"Sample Predictions from {results['split'].upper()} Set")
        print('='*80)
        
        predictions = results['predictions']
        
        # Best predictions
        print("\n🏆 BEST PREDICTIONS (Lowest WER):")
        print("-"*80)
        best_preds = sorted(predictions, key=lambda x: x['translation_wer'])[:num_samples]
        for i, pred in enumerate(best_preds, 1):
            print(f"\nSample {i} (WER: {pred['translation_wer']*100:.2f}%, BLEU: {pred['translation_bleu']*100:.2f}%)")
            print(f"GT Glosses:    {pred['gt_glosses']}")  # ADD THIS
            print(f"Pred Glosses:  {pred['pred_glosses']}")  # ADD THIS
            print(f"Gloss WER:     {pred['gloss_wer']*100:.2f}%")  # ADD THIS
            print(f"Ground Truth: {pred['gt_translation']}")
            print(f"Prediction:   {pred['pred_translation']}")
        
        # Worst predictions
        print(f"\n\n⚠️  WORST PREDICTIONS (Highest WER):")
        print("-"*80)
        worst_preds = sorted(predictions, key=lambda x: x['translation_wer'], reverse=True)[:num_samples]
        for i, pred in enumerate(worst_preds, 1):
            print(f"\nSample {i} (WER: {pred['translation_wer']*100:.2f}%, BLEU: {pred['translation_bleu']*100:.2f}%)")
            print(f"Ground Truth: {pred['gt_translation']}")
            print(f"Prediction:   {pred['pred_translation']}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained sign language model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: latest best_model.pth)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['test', 'dev'], help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    # Find latest checkpoint if not specified
    if args.checkpoint is None:
        exp_dirs = sorted(glob.glob('experiments/runs/exp_*'))
        if not exp_dirs:
            print("❌ Error: No experiments found!")
            print("   Please train a model first using: python scripts/05_train.py")
            return
        latest_exp = exp_dirs[-1]
        args.checkpoint = os.path.join(latest_exp, 'checkpoints', 'best_model.pth')
        print(f"Using latest checkpoint: {args.checkpoint}\n")
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint, args.config)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(split=args.split)
    
    # Print, save, and visualize results
    evaluator.print_results(results)
    evaluator.save_results(results)
    evaluator.visualize_results(results)
    evaluator.show_sample_predictions(results, num_samples=5)
    
    print(f"\n{'='*80}")
    print("✅ Evaluation Complete!")
    print(f"All results saved in: {evaluator.output_dir}")
    print('='*80)


if __name__ == '__main__':
    main()