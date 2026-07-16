import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MultimodalDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        # file_paths is a list of tuples: (eeg_path, eye_path, label)
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        eeg_path, eye_path, label = self.file_paths[idx]
        
        raw_eeg = np.load(eeg_path)
        raw_eye = np.load(eye_path)

        # Global static padding to 64, (time steps) for both modalities
        max_len = 64
        pad_eeg = max_len - raw_eeg.shape[1] # padding along time dimension(eeg)
        padded_eeg = np.pad(raw_eeg, ((0, 0), (0, pad_eeg), (0, 0)), mode='constant')
        
        pad_eye = max_len - raw_eye.shape[1] # padding along time dimension(eye)
        padded_eye = np.pad(raw_eye, ((0, 0), (0, pad_eye)), mode='constant')

        # Convert to PyTorch Channels-First format
        padded_eeg = np.transpose(padded_eeg, (2, 0, 1))
        padded_eye = np.expand_dims(padded_eye, axis=0)

        eeg_tensor = torch.tensor(padded_eeg, dtype=torch.float32)
        eye_tensor = torch.tensor(padded_eye, dtype=torch.float32)

        # DATA AUGMENTATION (Only active during Training)

        if self.augment:
            # 1. Gaussian Noise (50% chance): Prevents memorization of exact values
            if random.random() < 0.5:
                noise = torch.randn_like(eeg_tensor) * 0.05
                eeg_tensor = eeg_tensor + noise
            
            # 2. Temporal Masking (50% chance): Erases 5 continuous time steps 
            # to force the model to rely on the whole sequence context
            if random.random() < 0.5:
                mask_start = random.randint(0, max_len - 5)
                eeg_tensor[:, :, mask_start:mask_start+5] = 0
                eye_tensor[:, :, mask_start:mask_start+5] = 0

        return eeg_tensor, eye_tensor, torch.tensor(label, dtype=torch.long)
    
# why data augmentation :
# The dataset had limited variability,
# so the model could easily overfit by memorizing specific patterns.