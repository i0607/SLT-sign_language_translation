import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TrainingLogger:
    def __init__(self, log_dir):
        """Training logger with TensorBoard and matplotlib"""
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store metrics
        self.train_losses = []
        self.val_losses = []
        self.train_rec_losses = []
        self.train_trans_losses = []
        self.val_rec_losses = []
        self.val_trans_losses = []
        self.learning_rates = []
        self.epochs = []
    
    def log_epoch(self, epoch, train_loss, val_loss, 
                  train_rec_loss, train_trans_loss,
                  val_rec_loss, val_trans_loss, lr):
        """Log epoch metrics"""
        # Store
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_rec_losses.append(train_rec_loss)
        self.train_trans_losses.append(train_trans_loss)
        self.val_rec_losses.append(val_rec_loss)
        self.val_trans_losses.append(val_trans_loss)
        self.learning_rates.append(lr)
        
        # TensorBoard
        self.writer.add_scalars('Loss/Combined', {
            'train': train_loss, 'val': val_loss
        }, epoch)
        self.writer.add_scalars('Loss/Recognition', {
            'train': train_rec_loss, 'val': val_rec_loss
        }, epoch)
        self.writer.add_scalars('Loss/Translation', {
            'train': train_trans_loss, 'val': val_trans_loss
        }, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def create_loss_plots(self):
        """Create comprehensive loss plots"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combined Loss
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Combined Loss', fontsize=12)
        axes[0, 0].set_title('Combined Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recognition Loss
        axes[0, 1].plot(self.epochs, self.train_rec_losses, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(self.epochs, self.val_rec_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('CTC Loss')
        axes[0, 1].set_title('Recognition Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Translation Loss
        axes[1, 0].plot(self.epochs, self.train_trans_losses, 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(self.epochs, self.val_trans_losses, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CrossEntropy Loss')
        axes[1, 0].set_title('Translation Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
    def save_all_plots(self):
        """Generate all plots"""
        self.create_loss_plots()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()