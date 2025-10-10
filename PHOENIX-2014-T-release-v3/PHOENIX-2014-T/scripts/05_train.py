import sys
import os

# Fix Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
import torch
import pandas as pd

from src.data.dataset import ProperSignDataset
from src.data.vocabulary import build_vocabularies
from src.models.joint_model import SignLanguageTransformer
from src.training.trainer import Trainer

def main():
    # Load config
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading data and building vocabularies...")
    
    # Load training data
    train_data = pd.read_csv(config['data']['train_annotations'])
    
    # Build vocabularies
    gloss_vocab, word_vocab = build_vocabularies(train_data)
    print(f"Gloss vocabulary size: {len(gloss_vocab)}")
    print(f"Word vocabulary size: {len(word_vocab)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ProperSignDataset(
        config['data']['train_annotations'],
        config['data']['train_features'],
        gloss_vocab,
        word_vocab
    )
    
    val_dataset = ProperSignDataset(
        config['data']['dev_annotations'],
        config['data']['dev_features'],
        gloss_vocab,
        word_vocab
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = SignLanguageTransformer(
        gloss_vocab_size=len(gloss_vocab),
        word_vocab_size=len(word_vocab),
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer and train
    trainer = Trainer(model, train_dataset, val_dataset, config)
    trainer.train(num_epochs=config['training']['num_epochs'])

if __name__ == '__main__':
    main()