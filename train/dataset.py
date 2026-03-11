"""
Dataset class for loading butterfly images with location conditioning.
Implements Fourier positional encoding for smooth geographic representation.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional
import math


def fourier_encode_location(lat: float, lon: float, num_freqs: int = 32) -> np.ndarray:
    """
    Encode latitude/longitude using Fourier features.
    
    Maps continuous geographic coordinates to high-dimensional features that enable
    smooth interpolation and generalization to unseen locations.
    
    Args:
        lat: Latitude [-90, 90]
        lon: Longitude [-180, 180]
        num_freqs: Number of frequency bands (higher = more detail)
    
    Returns:
        features: Numpy array of shape (num_freqs * 2 * 2,) = (num_freqs * 4,)
                 Default: (128,) for num_freqs=32
    
    Reference: Tancik et al. "Fourier Features Let Networks Learn High Frequency 
               Functions in Low Dimensional Domains" (ICLR 2020)
    """
    # Normalize to [0, 1]
    lat_norm = (lat + 90.0) / 180.0  # [-90, 90] → [0, 1]
    lon_norm = (lon + 180.0) / 360.0  # [-180, 180] → [0, 1]
    
    coords = np.array([lat_norm, lon_norm])  # (2,)
    features = []
    
    # For each frequency band
    for k in range(num_freqs):
        # Frequency: 2^k
        freq = 2.0 ** k
        
        # For each coordinate (lat and lon)
        for coord in coords:
            # sin and cos at this frequency
            features.append(np.sin(freq * np.pi * coord))
            features.append(np.cos(freq * np.pi * coord))
    
    return np.array(features, dtype=np.float32)


def simple_encode_location(lat: float, lon: float) -> np.ndarray:
    """
    Simple normalization of coordinates to [-1, 1].
    
    Faster but less smooth than Fourier encoding.
    """
    lat_norm = lat / 90.0  # [-90, 90] → [-1, 1]
    lon_norm = lon / 180.0  # [-180, 180] → [-1, 1]
    return np.array([lat_norm, lon_norm], dtype=np.float32)


class ButterflyDataset(Dataset):
    """
    Dataset for butterfly images with location (lat/lon) conditioning.
    
    Loads images and their geographic coordinates from a CSV metadata file.
    Supports Fourier or simple location encoding.
    """
    
    def __init__(self, 
                 metadata_csv: str,
                 images_dir: str,
                 image_size: int = 512,
                 location_encoding: str = "fourier",
                 fourier_freqs: int = 32,
                 lat_column: str = "lat",
                 lon_column: str = "lon",
                 image_path_column: str = "image_path",
                 normalize_images: bool = True):
        """
        Args:
            metadata_csv: Path to CSV with metadata (lat, lon, image_path columns)
            images_dir: Base directory for images
            image_size: Target image size (will resize/crop to this)
            location_encoding: "fourier" or "simple"
            fourier_freqs: Number of Fourier frequencies (if using Fourier encoding)
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            image_path_column: Name of image path column
            normalize_images: Whether to normalize to [-1, 1] or keep [0, 1]
        """
        self.metadata_csv = metadata_csv
        self.images_dir = images_dir
        self.image_size = image_size
        self.location_encoding = location_encoding
        self.fourier_freqs = fourier_freqs
        self.normalize_images = normalize_images
        
        # Load metadata
        self.samples = []
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row[lat_column])
                    lon = float(row[lon_column])
                    img_path = row[image_path_column]
                    self.samples.append((lat, lon, img_path))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row {row} due to error: {e}")
        
        print(f"Loaded {len(self.samples)} samples from {metadata_csv}")
        
        # Validate image directory
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (3, image_size, image_size), normalized to [0, 1] or [-1, 1]
            location_emb: Tensor of shape (location_dim,), either 2 or 128 depending on encoding
        """
        lat, lon, img_path = self.samples[idx]
        
        # Load image
        full_img_path = os.path.join(self.images_dir, img_path)
        if not os.path.exists(full_img_path):
            # Try without "../data/images_licensed/" prefix if it's included
            full_img_path = img_path
        
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            # Return black image on failure
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor (0-1)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32)) / 255.0
        # Shape: (H, W, 3)
        
        # Normalize to [-1, 1] if requested
        if self.normalize_images:
            image_tensor = image_tensor * 2.0 - 1.0
        
        # Channel-first format
        image_tensor = image_tensor.permute(2, 0, 1)  # (3, H, W)
        
        # Encode location
        if self.location_encoding == "fourier":
            location_emb = fourier_encode_location(lat, lon, self.fourier_freqs)
        else:  # "simple"
            location_emb = simple_encode_location(lat, lon)
        
        location_emb = torch.from_numpy(location_emb)
        
        return image_tensor, location_emb


class ButterflyDataLoader:
    """Utility class to create train/val splits from ButterflyDataset."""
    
    @staticmethod
    def get_train_val_loaders(metadata_csv: str,
                              images_dir: str,
                              batch_size: int = 32,
                              train_split: float = 0.8,
                              num_workers: int = 4,
                              pin_memory: bool = True,
                              seed: int = 42,
                              **dataset_kwargs) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create train and validation dataloaders.
        
        Args:
            metadata_csv: Path to metadata CSV
            images_dir: Directory with images
            batch_size: Batch size for dataloaders
            train_split: Fraction for training (0-1)
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            seed: Random seed for reproducibility
            **dataset_kwargs: Additional arguments for ButterflyDataset
        
        Returns:
            train_loader, val_loader
        """
        # Create full dataset
        dataset = ButterflyDataset(metadata_csv, images_dir, **dataset_kwargs)
        
        # Train/val split
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from config import get_config
    
    config = get_config()
    
    try:
        dataset = ButterflyDataset(
            metadata_csv=config.metadata_csv,
            images_dir=config.images_dir,
            image_size=config.image_size,
            location_encoding=config.location_encoding,
            fourier_freqs=config.fourier_freqs
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test sample
        image, location_emb = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Location embedding shape: {location_emb.shape}")
        print(f"Location embedding sample: {location_emb[:10]}")
        
    except FileNotFoundError as e:
        print(f"Note: {e}")
        print("This is expected if data hasn't been downloaded yet.")
