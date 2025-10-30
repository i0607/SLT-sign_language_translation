import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TrainingLogger:
    def __init__(self, log_dir, config=None):
        """
        Training logger with TensorBoard and matplotlib
        
        Args:
            log_dir: Directory to save logs
            config: Configuration dictionary to include in plots
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store configuration
        self.config = config
        
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
    
    def _get_config_text(self):
        """Extract key configuration parameters as formatted text"""
        if self.config is None:
            return "No configuration available"
        
        config_lines = []
        
        # Model parameters
        if 'model' in self.config:
            model = self.config['model']
            config_lines.append("Model Architecture:")
            config_lines.append(f"  Hidden Dim: {model.get('hidden_dim', 'N/A')}")
            config_lines.append(f"  Layers: {model.get('num_layers', 'N/A')}")
            config_lines.append(f"  Heads: {model.get('num_heads', 'N/A')}")
            config_lines.append(f"  Dropout: {model.get('dropout', 'N/A')}")
        
        # Training parameters
        if 'training' in self.config:
            train = self.config['training']
            config_lines.append("\nTraining Settings:")
            config_lines.append(f"  Batch Size: {train.get('batch_size', 'N/A')}")
            config_lines.append(f"  Initial LR: {train.get('learning_rate', 'N/A')}")
            config_lines.append(f"  λ_recognition: {train.get('lambda_recognition', 'N/A')}")
            config_lines.append(f"  λ_translation: {train.get('lambda_translation', 'N/A')}")
            config_lines.append(f"  Grad Clip: {train.get('gradient_clip', 'N/A')}")
            config_lines.append(f"  Weight Decay: {train.get('weight_decay', 'N/A')}")
            config_lines.append(f"  Scheduler Factor: {train.get('scheduler_factor', 'N/A')}")
            config_lines.append(f"  Scheduler Patience: {train.get('scheduler_patience', 'N/A')}")
        
        return '\n'.join(config_lines)
    
    def _get_config_title(self):
        """Get concise config summary for plot title"""
        if self.config is None:
            return ""
        
        title_parts = []
        
        if 'model' in self.config:
            model = self.config['model']
            title_parts.append(f"hidden={model.get('hidden_dim', '?')}")
            title_parts.append(f"layers={model.get('num_layers', '?')}")
            title_parts.append(f"heads={model.get('num_heads', '?')}")
        
        if 'training' in self.config:
            train = self.config['training']
            title_parts.append(f"bs={train.get('batch_size', '?')}")
            title_parts.append(f"lr={train.get('learning_rate', '?')}")
            title_parts.append(f"λr={train.get('lambda_recognition', '?')}")
            title_parts.append(f"λt={train.get('lambda_translation', '?')}")
        
        return " | ".join(title_parts)
    
    def create_loss_plots(self):
        """Create comprehensive loss plots with configuration details"""
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 11))
        
        # Create a grid with space for config text
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, 
                             top=0.92, bottom=0.08, left=0.08, right=0.75)
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Overall title with config summary
        config_summary = self._get_config_title()
        fig.suptitle(f'Training Progress\n{config_summary}', 
                    fontsize=13, fontweight='bold', y=0.98)
        
        # Combined Loss
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Combined Loss', fontsize=11)
        ax1.set_title('Combined Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Recognition Loss (CTC)
        ax2.plot(self.epochs, self.train_rec_losses, 'b-', label='Train', linewidth=2)
        ax2.plot(self.epochs, self.val_rec_losses, 'r-', label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('CTC Loss', fontsize=11)
        ax2.set_title('Recognition Loss (CTC)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Translation Loss
        ax3.plot(self.epochs, self.train_trans_losses, 'b-', label='Train', linewidth=2)
        ax3.plot(self.epochs, self.val_trans_losses, 'r-', label='Validation', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('CrossEntropy Loss', fontsize=11)
        ax3.set_title('Translation Loss', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate Schedule
        ax4.plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Learning Rate', fontsize=11)
        ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Add configuration text box on the right side
        config_text = self._get_config_text()
        
        # Create text box axis
        ax_text = fig.add_subplot(gs[:, 1])
        ax_text.axis('off')
        
        # Add text with background box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.15)
        ax_text.text(1.15, 0.5, config_text, 
                    transform=ax_text.transAxes,
                    fontsize=9,
                    verticalalignment='center',
                    fontfamily='monospace',
                    bbox=props)
        
        # Add training statistics
        if len(self.epochs) > 0:
            stats_text = "Training Statistics:\n"
            stats_text += f"  Total Epochs: {self.epochs[-1]}\n"
            stats_text += f"  Best Val Loss: {min(self.val_losses):.4f}\n"
            stats_text += f"  Final Train Loss: {self.train_losses[-1]:.4f}\n"
            stats_text += f"  Final Val Loss: {self.val_losses[-1]:.4f}\n"
            stats_text += f"  Final LR: {self.learning_rates[-1]:.2e}"
            
            ax_text.text(1.15, 0.05, stats_text,
                        transform=ax_text.transAxes,
                        fontsize=9,
                        verticalalignment='bottom',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.15))
        
        # Save figure
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training curves with configuration details to {self.plots_dir}")
    
    def create_individual_plots(self):
        """Create individual plots for each metric with config details"""
        
        config_text = self._get_config_text()
        metrics = [
            ('Combined Loss', self.train_losses, self.val_losses, 'combined_loss.png'),
            ('Recognition Loss', self.train_rec_losses, self.val_rec_losses, 'recognition_loss.png'),
            ('Translation Loss', self.train_trans_losses, self.val_trans_losses, 'translation_loss.png'),
        ]
        
        for title, train_data, val_data, filename in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot data
            ax.plot(self.epochs, train_data, 'b-', label='Train', linewidth=2)
            ax.plot(self.epochs, val_data, 'r-', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{title}\n{self._get_config_title()}', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add config text
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.1)
            ax.text(1.02, 0.5, config_text,
                   transform=ax.transAxes,
                   fontsize=8,
                   verticalalignment='center',
                   fontfamily='monospace',
                   bbox=props)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, filename), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_all_plots(self):
        """Generate all plots"""
        self.create_loss_plots()
        self.create_individual_plots()
        print(f"✓ All plots saved to {self.plots_dir}")
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()