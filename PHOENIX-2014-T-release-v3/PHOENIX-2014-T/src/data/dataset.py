import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

try:
    import imageio.v2 as imageio
    _use_imageio = True
except Exception:
    from PIL import Image
    _use_imageio = False

class ProperSignDataset(Dataset):
    def __init__(self, csv_file, feature_dir, gloss_vocab, word_vocab, max_frames=32):
        """
        Sign Language Dataset
        
        Args:
            csv_file: Path to CSV with annotations
            feature_dir: Directory containing video frame folders
            gloss_vocab: Dictionary mapping glosses to IDs
            word_vocab: Dictionary mapping words to IDs
            max_frames: Maximum number of frames to load per video
        """
        self.data = pd.read_csv(csv_file)
        self.feature_dir = feature_dir
        self.gloss_vocab = gloss_vocab
        self.word_vocab = word_vocab
        self.max_frames = max_frames
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load video frames (directory of PNGs)
        video_name = str(row['name']).split('.')[0]
        frames_dir = os.path.join(self.feature_dir, video_name)
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        
        if not frame_paths:
            raise FileNotFoundError(f'No frames found in {frames_dir}')
        
        # Subsample frames to max_frames
        sample_paths = frame_paths[:self.max_frames]
        
        # Load frames
        if _use_imageio:
            frames = np.stack([imageio.imread(p) for p in sample_paths], axis=0)
        else:
            frames = np.stack([np.array(Image.open(p)) for p in sample_paths], axis=0)
        
        # Convert glosses to IDs
        glosses = str(row['orth_english']).split()
        gloss_ids = [self.gloss_vocab.get(g, 0) for g in glosses]
        
        # Convert words to IDs
        words = str(row['translation_english']).split()
        word_ids = [self.word_vocab['<SOS>']] + \
                   [self.word_vocab.get(w, self.word_vocab['<UNK>']) for w in words] + \
                   [self.word_vocab['<EOS>']]
        
        return {
            'frames': torch.FloatTensor(frames),
            'gloss_ids': torch.LongTensor(gloss_ids),
            'word_ids': torch.LongTensor(word_ids)
        }