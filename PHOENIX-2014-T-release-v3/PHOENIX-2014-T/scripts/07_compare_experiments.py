"""
Compare Multiple Training Experiments
Generates comparison tables and plots across different experimental runs

Usage:
    python scripts/07_compare_experiments.py
    python scripts/07_compare_experiments.py --output comparison_report.png
"""

import sys
import os
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Fix Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_experiment_results(exp_dir):
    """Load results from an experiment directory"""
    exp_name = os.path.basename(exp_dir)
    
    # Load config
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load evaluation results
    eval_dir = os.path.join(exp_dir, 'evaluation')
    
    results = {
        'exp_name': exp_name,
        'exp_dir': exp_dir,
        'timestamp': exp_name.split('_', 1)[1] if '_' in exp_name else 'unknown',
        
        # Model config
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'num_heads': config['model']['num_heads'],
        'dropout': config['model']['dropout'],
        
        # Training config
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs'],
        'lambda_recognition': config['training']['lambda_recognition'],
        'lambda_translation': config['training']['lambda_translation'],
        'gradient_clip': config['training']['gradient_clip'],
        'weight_decay': config['training']['weight_decay'],
    }
    
    # Load test results if available
    test_results_path = os.path.join(eval_dir, 'test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        results.update({
            'test_loss': test_results.get('avg_loss'),
            'test_rec_loss': test_results.get('avg_recognition_loss'),
            'test_trans_loss': test_results.get('avg_translation_loss'),
            'test_gloss_wer': test_results.get('gloss_wer'),
            'test_trans_wer': test_results.get('translation_wer'),
            'test_bleu': test_results.get('translation_bleu'),
        })
    
    # Load dev results if available
    dev_results_path = os.path.join(eval_dir, 'dev_results.json')
    if os.path.exists(dev_results_path):
        with open(dev_results_path, 'r') as f:
            dev_results = json.load(f)
        results.update({
            'dev_loss': dev_results.get('avg_loss'),
            'dev_rec_loss': dev_results.get('avg_recognition_loss'),
            'dev_trans_loss': dev_results.get('avg_translation_loss'),
            'dev_gloss_wer': dev_results.get('gloss_wer'),
            'dev_trans_wer': dev_results.get('translation_wer'),
            'dev_bleu': dev_results.get('translation_bleu'),
        })
    
    # Load checkpoint info
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    if os.path.exists(checkpoint_path):
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        results['best_val_loss'] = checkpoint.get('best_val_loss')
        results['trained_epochs'] = checkpoint.get('epoch')
    
    return results


def compare_experiments(output_path='experiment_comparison.png'):
    """Compare all experiments and generate visualizations"""
    
    print("\n" + "="*80)
    print("Comparing Training Experiments")
    print("="*80 + "\n")
    
    # Find all experiments
    exp_dirs = sorted(glob.glob('experiments/runs/exp_*'))
    if not exp_dirs:
        print("❌ No experiments found!")
        return
    
    print(f"Found {len(exp_dirs)} experiments\n")
    
    # Load all experiment results
    all_results = []
    for exp_dir in exp_dirs:
        results = load_experiment_results(exp_dir)
        if results:
            all_results.append(results)
        else:
            print(f"⚠️  Skipping {os.path.basename(exp_dir)} (missing config)")
    
    if not all_results:
        print("❌ No valid experiments with results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print comparison table
    print("="*120)
    print("Experiment Comparison Table")
    print("="*120)
    
    # Select key columns for display
    display_cols = [
        'exp_name', 'hidden_dim', 'num_layers', 'batch_size', 'learning_rate',
        'lambda_recognition', 'lambda_translation', 'trained_epochs',
        'test_trans_wer', 'test_bleu', 'best_val_loss'
    ]
    
    # Filter to available columns
    display_cols = [col for col in display_cols if col in df.columns]
    display_df = df[display_cols].copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print("="*120 + "\n")
    
    # Find best experiment
    if 'test_trans_wer' in df.columns and df['test_trans_wer'].notna().any():
        best_idx = df['test_trans_wer'].idxmin()
        best_exp = df.loc[best_idx]
        
        print("🏆 BEST EXPERIMENT (Lowest Translation WER):")
        print("-"*80)
        print(f"   Name: {best_exp['exp_name']}")
        print(f"   Translation WER: {best_exp['test_trans_wer']:.2f}%")
        print(f"   BLEU Score: {best_exp['test_bleu']:.2f}%")
        print(f"   Configuration:")
        print(f"      • Hidden Dim: {best_exp['hidden_dim']}")
        print(f"      • Layers: {best_exp['num_layers']}")
        print(f"      • Learning Rate: {best_exp['learning_rate']}")
        print(f"      • Batch Size: {best_exp['batch_size']}")
        print(f"      • Lambda Recognition: {best_exp['lambda_recognition']}")
        print(f"      • Lambda Translation: {best_exp['lambda_translation']}")
        print()
    
    # Generate visualizations if we have test results
    if 'test_trans_wer' in df.columns and df['test_trans_wer'].notna().sum() > 1:
        print("Generating comparison visualizations...\n")
        create_comparison_plots(df, output_path)
        print(f"✓ Saved comparison plots to: {output_path}\n")
    
    # Save comparison table
    csv_path = 'experiment_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table to: {csv_path}\n")
    
    print("="*80)
    print("✅ Comparison Complete!")
    print("="*80 + "\n")


def create_comparison_plots(df, output_path):
    """Create comparison visualization plots"""
    
    # Filter experiments with test results
    df_test = df[df['test_trans_wer'].notna()].copy()
    
    if len(df_test) < 2:
        print("Need at least 2 experiments with test results for plots")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Add experiment labels
    df_test['exp_label'] = [f"Exp {i+1}" for i in range(len(df_test))]
    
    # 1. Translation WER by Experiment
    axes[0, 0].bar(range(len(df_test)), df_test['test_trans_wer'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Experiment')
    axes[0, 0].set_ylabel('Translation WER (%)')
    axes[0, 0].set_title('Translation WER by Experiment')
    axes[0, 0].set_xticks(range(len(df_test)))
    axes[0, 0].set_xticklabels(df_test['exp_label'], rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(df_test['test_trans_wer']):
        axes[0, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. BLEU Score by Experiment
    axes[0, 1].bar(range(len(df_test)), df_test['test_bleu'], color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Experiment')
    axes[0, 1].set_ylabel('BLEU Score (%)')
    axes[0, 1].set_title('BLEU Score by Experiment')
    axes[0, 1].set_xticks(range(len(df_test)))
    axes[0, 1].set_xticklabels(df_test['exp_label'], rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(df_test['test_bleu']):
        axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Learning Rate vs Performance
    if len(df_test['learning_rate'].unique()) > 1:
        axes[0, 2].scatter(df_test['learning_rate'], df_test['test_trans_wer'], 
                          s=100, c='orange', alpha=0.7, edgecolors='black')
        axes[0, 2].set_xlabel('Learning Rate')
        axes[0, 2].set_ylabel('Translation WER (%)')
        axes[0, 2].set_title('Learning Rate vs WER')
        axes[0, 2].set_xscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'All experiments\nuse same LR', 
                       ha='center', va='center', fontsize=12)
        axes[0, 2].set_title('Learning Rate vs WER')
        axes[0, 2].axis('off')
    
    # 4. Batch Size vs Performance
    if len(df_test['batch_size'].unique()) > 1:
        axes[1, 0].scatter(df_test['batch_size'], df_test['test_trans_wer'], 
                          s=100, c='purple', alpha=0.7, edgecolors='black')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Translation WER (%)')
        axes[1, 0].set_title('Batch Size vs WER')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'All experiments\nuse same batch size', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Batch Size vs WER')
        axes[1, 0].axis('off')
    
    # 5. Lambda Translation vs BLEU
    if len(df_test['lambda_translation'].unique()) > 1:
        axes[1, 1].scatter(df_test['lambda_translation'], df_test['test_bleu'], 
                          s=100, c='red', alpha=0.7, edgecolors='black')
        axes[1, 1].set_xlabel('Lambda Translation')
        axes[1, 1].set_ylabel('BLEU Score (%)')
        axes[1, 1].set_title('Lambda Translation vs BLEU')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'All experiments\nuse same λ_translation', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Lambda Translation vs BLEU')
        axes[1, 1].axis('off')
    
    # 6. Performance Summary
    metrics = ['WER', 'BLEU', 'Gloss WER']
    best_exp_idx = df_test['test_trans_wer'].idxmin()
    best_values = [
        df_test.loc[best_exp_idx, 'test_trans_wer'],
        df_test.loc[best_exp_idx, 'test_bleu'],
        df_test.loc[best_exp_idx, 'test_gloss_wer']
    ]
    
    avg_values = [
        df_test['test_trans_wer'].mean(),
        df_test['test_bleu'].mean(),
        df_test['test_gloss_wer'].mean()
    ]
    
    x = range(len(metrics))
    width = 0.35
    
    axes[1, 2].bar([i - width/2 for i in x], best_values, width, 
                   label='Best', color='gold', alpha=0.7, edgecolor='black')
    axes[1, 2].bar([i + width/2 for i in x], avg_values, width,
                   label='Average', color='gray', alpha=0.7, edgecolor='black')
    
    axes[1, 2].set_ylabel('Score (%)')
    axes[1, 2].set_title('Best vs Average Performance')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description='Compare training experiments')
    parser.add_argument('--output', type=str, default='experiment_comparison.png',
                       help='Output path for comparison plots')
    
    args = parser.parse_args()
    
    compare_experiments(args.output)


if __name__ == '__main__':
    main()