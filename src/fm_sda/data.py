
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

class SpectralDataset(Dataset):
    def __init__(
        self, 
        csv_path: str, 
        spectral_columns_start: int = 4, # Based on data inspection
        apply_derivative: bool = True,
        window_length: int = 5,
        polyorder: int = 2,
        deriv: int = 1,
        normalize: bool = True
    ):
        """
        Dataset for Spectral Data.
        Assumes metadata in first few columns, followed by spectral data.
        """
        self.data = pd.read_csv(csv_path)
        
        # Extract features (spectra) and labels (metadata)
        self.spectral_data = self.data.iloc[:, spectral_columns_start:].values.astype(np.float32)
        self.metadata = self.data.iloc[:, :spectral_columns_start]
        
        # Preprocessing: Derivative
        if apply_derivative:
            self.spectral_data = savgol_filter(
                self.spectral_data, 
                window_length=window_length, 
                polyorder=polyorder, 
                deriv=deriv, 
                axis=1
            )
            
        # Normalization
        self.scaler = StandardScaler()
        if normalize:
            self.spectral_data = self.scaler.fit_transform(self.spectral_data)
            
        # Label Encoding for 'Giong' (Variety)
        self.label_encoder = LabelEncoder()
        if 'Giong' in self.metadata.columns:
            self.labels = self.label_encoder.fit_transform(self.metadata['Giong'])
        else:
            print("Warning: 'Giong' column not found. creating dummy labels.")
            self.labels = np.zeros(len(self.data), dtype=int)
            
        self.spectral_data = torch.tensor(self.spectral_data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectral_data[idx], self.labels[idx]

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        return self.scaler.inverse_transform(data)

    @property
    def input_dim(self) -> int:
        return self.spectral_data.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)
