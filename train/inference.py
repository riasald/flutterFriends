"""
Inference script for CVAE: Generate butterfly images given locations.
Can also interpolate between locations and test the trained model.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from config import get_config
from models import CVAE
from dataset import fourier_encode_location, simple_encode_location


class CVAEInference:
    """Inference utility for trained CVAE model."""
    
    def __init__(self, checkpoint_path: str, config=None, device: str = "cuda"):
        """
        Load trained CVAE model.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            config: CVAEConfig object (optional, loaded from checkpoint if not provided)
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config or get_config()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = CVAE(
            encoder_channels=self.config.encoder_channels,
            decoder_channels=self.config.decoder_channels,
            location_dim=self.config.location_dim,
            latent_dim=self.config.latent_dim
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Model on {self.device}")
    
    def encode_location(self, lat: float, lon: float) -> torch.Tensor:
        """Encode location to embedding."""
        if self.config.location_encoding == "fourier":
            location_emb = fourier_encode_location(lat, lon, self.config.fourier_freqs)
        else:
            location_emb = simple_encode_location(lat, lon)
        
        return torch.from_numpy(location_emb).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def generate_image(self, lat: float, lon: float, seed: int = None) -> np.ndarray:
        """
        Generate a single butterfly image for given location.
        
        Args:
            lat: Latitude [-90, 90]
            lon: Longitude [-180, 180]
            seed: Random seed for reproducibility (optional)
        
        Returns:
            image: Numpy array (512, 512, 3) in [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encode location
        location_emb = self.encode_location(lat, lon)
        
        # Sample from prior
        z = torch.randn(1, self.config.latent_dim, device=self.device)
        
        # Decode
        image = self.model.decode(z, location_emb)  # (1, 3, 512, 512)
        
        # Convert to numpy and permute to HWC
        image = image.squeeze(0).cpu().numpy()  # (3, 512, 512)
        image = np.transpose(image, (1, 2, 0))  # (512, 512, 3)
        image = np.clip(image, 0, 1)  # Ensure in [0, 1]
        
        return image
    
    @torch.no_grad()
    def generate_variations(self, lat: float, lon: float, 
                          num_samples: int = 8) -> np.ndarray:
        """
        Generate multiple variations of a butterfly at given location.
        
        Returns:
            images: Array of shape (num_samples, 512, 512, 3)
        """
        location_emb = self.encode_location(lat, lon)
        location_emb = location_emb.repeat(num_samples, 1)  # (num_samples, location_dim)
        
        # Sample from prior
        z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
        
        # Decode
        images = self.model.decode(z, location_emb)  # (num_samples, 3, 512, 512)
        
        # Convert to numpy
        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # (num_samples, 512, 512, 3)
        images = np.clip(images, 0, 1)
        
        return images
    
    @torch.no_grad()
    def interpolate_locations(self, lat1: float, lon1: float,
                            lat2: float, lon2: float,
                            num_steps: int = 5, seed: int = None) -> np.ndarray:
        """
        Interpolate between two locations with fixed latent code.
        Shows geographic gradience in appearance.
        
        Args:
            lat1, lon1: Start location
            lat2, lon2: End location
            num_steps: Number of interpolation steps
            seed: Reproducibility seed
        
        Returns:
            images: Array of shape (num_steps, 512, 512, 3)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Sample latent code (constant across interpolation)
        z = torch.randn(1, self.config.latent_dim, device=self.device)
        
        images = []
        for t in np.linspace(0, 1, num_steps):
            # Interpolate location
            lat = lat1 * (1 - t) + lat2 * t
            lon = lon1 * (1 - t) + lon2 * t
            
            # Encode
            location_emb = self.encode_location(lat, lon)
            
            # Decode with fixed z
            image = self.model.decode(z, location_emb)
            image = image.squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = np.clip(image, 0, 1)
            
            images.append(image)
        
        return np.array(images)
    
    def visualize_variations(self, lat: float, lon: float, num_samples: int = 8,
                           save_path: str = None):
        """Visualize variations for a location."""
        images = self.generate_variations(lat, lon, num_samples)
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < num_samples:
                ax.imshow(images[i])
                ax.set_title(f"Variation {i+1}")
            ax.axis('off')
        
        plt.suptitle(f"Butterfly Variations at Lat={lat:.2f}, Lon={lon:.2f}")
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def visualize_interpolation(self, lat1: float, lon1: float,
                               lat2: float, lon2: float,
                               save_path: str = None):
        """Visualize interpolation between two locations."""
        images = self.interpolate_locations(lat1, lon1, lat2, lon2, num_steps=7)
        
        fig = plt.figure(figsize=(15, 3))
        
        for i, image in enumerate(images):
            ax = plt.subplot(1, len(images), i + 1)
            ax.imshow(image)
            ax.axis('off')
        
        plt.suptitle(f"Geographic Interpolation: "
                    f"({lat1:.1f},{lon1:.1f}) → ({lat2:.1f},{lon2:.1f})")
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()


def main():
    """Example inference."""
    config = get_config()
    
    # Path to trained checkpoint
    checkpoint_path = Path(config.output_dir) / f"{config.checkpoint_name}_best.pt"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first.")
        return
    
    # Initialize inference
    infer = CVAEInference(str(checkpoint_path), config)
    
    # Generate butterflies at test locations (from config)
    print("\nGenerating butterflies at test locations...\n")
    
    for lat, lon in config.validation_locations[:3]:
        print(f"Location: ({lat:.2f}, {lon:.2f})")
        
        # Generate single image
        image = infer.generate_image(lat, lon)
        
        # Visualize variations
        infer.visualize_variations(lat, lon, num_samples=8,
                                   save_path=f"butterfly_{lat:.1f}_{lon:.1f}.png")
    
    # Test interpolation
    print("\nTesting geographic interpolation...")
    infer.visualize_interpolation(25.76, -80.19, 34.05, -118.24,
                                 save_path="interpolation_florida_to_la.png")


if __name__ == "__main__":
    main()
