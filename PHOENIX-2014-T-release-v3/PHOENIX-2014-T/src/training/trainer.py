import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import json
from datetime import datetime

from src.training.losses import JointLoss
from src.utils.logging import TrainingLogger

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Trainer for Sign Language Transformers
        
        Args:
            model: SignLanguageTransformer model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Device
        device_name = config['training']['device']
        if device_name == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            print(f"Warning: Requested device '{device_name}' not available, using CPU")
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4),
            collate_fn=self.collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4),
            collate_fn=self.collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Loss function
        self.criterion = JointLoss(
            lambda_r=config['training']['lambda_recognition'],
            lambda_t=config['training']['lambda_translation'],
            blank_idx=0,
            pad_idx=-100
        )
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.9, 0.998),
            weight_decay=config['training'].get('weight_decay', 1e-3)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training'].get('scheduler_factor', 0.7),
            patience=config['training'].get('scheduler_patience', 8),
            verbose=True,
            min_lr=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join('experiments', 'runs', f'exp_{timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(os.path.join(self.experiment_dir, 'logs'))
        
        # Save config
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def collate_fn(self, batch):
        """
        Collate function to handle variable-length sequences
        Pads all sequences to the maximum length in the batch
        
        Args:
            batch: List of dataset items, each containing:
                   - frames: (num_frames, height, width, channels)
                   - gloss_ids: (num_glosses,)
                   - word_ids: (num_words,)
        
        Returns:
            Dictionary with batched and padded tensors:
                - frames: (batch_size, max_frames, height, width, channels)
                - glosses: (batch_size, max_glosses) - padded with 0
                - words: (batch_size, max_words) - padded with -100
                - input_lengths: (batch_size,) - actual frame counts
                - gloss_lengths: (batch_size,) - actual gloss counts
        """
        # Find max lengths in this batch
        max_frames = max(item['frames'].shape[0] for item in batch)
        max_glosses = max(len(item['gloss_ids']) for item in batch)
        max_words = max(len(item['word_ids']) for item in batch)
        
        batch_size = len(batch)
        
        # Get dimensions from first item
        _, height, width, channels = batch[0]['frames'].shape
        
        # Initialize tensors with appropriate padding values
        frames_batch = torch.zeros(batch_size, max_frames, height, width, channels, dtype=torch.float32)
        glosses_batch = torch.zeros(batch_size, max_glosses, dtype=torch.long)  # Pad with 0 (blank token)
        words_batch = torch.full((batch_size, max_words), -100, dtype=torch.long)  # Pad with -100 (ignore index)
        
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        gloss_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            # Frames - copy actual frames
            num_frames = item['frames'].shape[0]
            frames_batch[i, :num_frames] = item['frames']
            input_lengths[i] = num_frames
            
            # Glosses - copy actual glosses (rest stay 0)
            num_glosses = len(item['gloss_ids'])
            glosses_batch[i, :num_glosses] = item['gloss_ids']
            gloss_lengths[i] = num_glosses
            
            # Words - copy actual words (rest stay -100)
            num_words = len(item['word_ids'])
            words_batch[i, :num_words] = item['word_ids']
        
        return {
            'frames': frames_batch,
            'glosses': glosses_batch,
            'words': words_batch,
            'input_lengths': input_lengths,
            'gloss_lengths': gloss_lengths
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_rec_loss = 0
        total_trans_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                frames = batch['frames'].to(self.device)
                glosses = batch['glosses'].to(self.device)
                words = batch['words'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                gloss_lengths = batch['gloss_lengths'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
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
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('gradient_clip', 1.0)
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
                total_trans_loss += trans_loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rec': f"{rec_loss.item():.4f}",
                    'trans': f"{trans_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            raise RuntimeError("No valid batches in training epoch")
        
        avg_loss = total_loss / num_batches
        avg_rec_loss = total_rec_loss / num_batches
        avg_trans_loss = total_trans_loss / num_batches
        
        return avg_loss, avg_rec_loss, avg_trans_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_rec_loss = 0
        total_trans_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]"):
                try:
                    # Move to device
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
                        
                except RuntimeError as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if num_batches == 0:
            raise RuntimeError("No valid batches in validation")
        
        avg_loss = total_loss / num_batches
        avg_rec_loss = total_rec_loss / num_batches
        avg_trans_loss = total_trans_loss / num_batches
        
        return avg_loss, avg_rec_loss, avg_trans_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.experiment_dir,
            'checkpoints',
            f'checkpoint_epoch_{self.current_epoch:03d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.experiment_dir,
                'checkpoints',
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print("=" * 80)
        print(f"Starting Training")
        print("=" * 80)
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Lambda Recognition: {self.config['training']['lambda_recognition']}")
        print(f"Lambda Translation: {self.config['training']['lambda_translation']}")
        print("=" * 80)
        print(f"\nTo view TensorBoard:")
        print(f"  tensorboard --logdir={os.path.join(self.experiment_dir, 'logs')}")
        print(f"  Then open: http://localhost:6006\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            try:
                # Train
                train_loss, train_rec_loss, train_trans_loss = self.train_epoch()
                
                # Validate
                val_loss, val_rec_loss, val_trans_loss = self.validate()
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.logger.log_epoch(
                    epoch + 1,
                    train_loss, val_loss,
                    train_rec_loss, train_trans_loss,
                    val_rec_loss, val_trans_loss,
                    current_lr
                )
                
                # Print epoch summary
                print("\n" + "=" * 80)
                print(f"Epoch {epoch+1}/{num_epochs} Summary:")
                print("-" * 80)
                print(f"  Train Loss: {train_loss:.4f} (Rec: {train_rec_loss:.4f}, Trans: {train_trans_loss:.4f})")
                print(f"  Val Loss:   {val_loss:.4f} (Rec: {val_rec_loss:.4f}, Trans: {val_trans_loss:.4f})")
                print(f"  LR: {current_lr:.6f}")
                print("=" * 80 + "\n")
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(is_best=is_best)
                
                # Generate plots every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.logger.save_all_plots()
                
                # Early stopping check
                if current_lr < 1e-6:
                    print("Learning rate too small. Stopping training.")
                    break
                    
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                raise
        
        # Final plots and cleanup
        print("\nGenerating final visualizations...")
        self.logger.save_all_plots()
        self.logger.close()
        
        print("\n" + "=" * 80)
        print("Training Completed!")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Results saved in: {self.experiment_dir}")
        print(f"  - Checkpoints: {os.path.join(self.experiment_dir, 'checkpoints')}")
        print(f"  - Plots: {os.path.join(self.experiment_dir, 'logs', 'plots')}")
        print(f"  - TensorBoard logs: {os.path.join(self.experiment_dir, 'logs')}")
        print("=" * 80)