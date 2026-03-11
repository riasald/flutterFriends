"""
Configuration for CVAE training on butterfly location-conditioned image generation.
All hyperparameters centralized for easy tuning.
"""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CVAEConfig:
    """CVAE training hyperparameters."""
    
    # ===== DEVICE & HARDWARE =====
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 8  # DataLoader workers
    pin_memory: bool = True
    
    # ===== DATA =====
    data_dir: str = "../data"
    metadata_csv: str = "../data/filtered_metadata/metadata_quality.csv"
    images_dir: str = "../data/images_licensed"
    image_size: int = 512  # Target image resolution
    
    # Metadata columns (adjust if schema differs)
    lat_column: str = "lat"
    lon_column: str = "lon"
    image_path_column: str = "image_path"
    
    # Data split
    train_split: float = 0.8  # 80% train, 20% val
    seed: int = 42
    
    # ===== LOCATION ENCODING =====
    location_encoding: str = "fourier"  # "fourier" or "simple"
    fourier_freqs: int = 32  # Number of frequencies for Fourier encoding
    # Fourier output dimension: fourier_freqs * 2 * 2 = 128D (32 freqs × sin/cos × lat/lon)
    
    # ===== MODEL ARCHITECTURE =====
    # Encoder channels: input → bottleneck
    encoder_channels: Tuple[int, ...] = (3, 16, 32, 64, 64)
    
    # Decoder channels: bottleneck → output
    decoder_channels: Tuple[int, ...] = (64, 64, 32, 16, 3)
    
    # Latent dimension (stores butterfly appearance variation + noise)
    latent_dim: int = 128
    
    # Location embedding dimension (after Fourier encoding)
    location_dim: int = 128  # fourier_freqs * 2 * 2
    
    # ===== TRAINING HYPERPARAMETERS =====
    batch_size: int = 32  # Per-batch size (cloud GPU: can handle 32+)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    epochs: int = 50  # Full convergence on cloud GPU
    checkpoint_interval: int = 5  # Save checkpoint every N epochs
    
    # ===== KL DIVERGENCE SCHEDULING (CRITICAL FOR POSTERIOR COLLAPSE) =====
    beta_init: float = 0.0  # Start at 0 (minimize KL initially)
    beta_max: float = 1.0  # Target KL weight
    beta_warmup_epochs: int = 10  # Linearly increase β over first 10 epochs
    # After warmup, β stays at 1.0
    
    # ===== LOSS FUNCTION =====
    reconstruction_loss: str = "mse"  # "mse" or "l1"
    
    # ===== OUTPUTS & CHECKPOINTING =====
    output_dir: str = "../models"
    checkpoint_name: str = "cvae_butterfly_512x512"
    log_dir: str = "../logs"
    
    # ===== SAMPLING & VALIDATION =====
    num_samples_per_location: int = 8  # Generate N butterflies per location during validation
    validation_locations: list = None  # Will be set to [(lat, lon), ...] for validation
    
    # Default validation locations (spread across US)
    def __post_init__(self):
        if self.validation_locations is None:
            # Geographic spread: Florida, Texas, California, New York, Colorado
            self.validation_locations = [
                (25.76, -80.19),   # Miami, FL
                (29.76, -95.37),   # Houston, TX
                (34.05, -118.24),  # Los Angeles, CA
                (40.71, -74.01),   # New York, NY
                (39.74, -104.99),  # Denver, CO
                (37.77, -122.42),  # San Francisco, CA
            ]


def get_config() -> CVAEConfig:
    """Load configuration (can be overridden by env vars or command line)."""
    return CVAEConfig()


if __name__ == "__main__":
    config = get_config()
    print("CONFIG:")
    print(f"  Device: {config.device}")
    print(f"  Image size: {config.image_size}×{config.image_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Location encoding: {config.location_encoding} (dim={config.location_dim})")
    print(f"  Epochs: {config.epochs}")
    print(f"  Beta schedule: {config.beta_init}→{config.beta_max} over {config.beta_warmup_epochs} epochs")
