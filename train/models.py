"""
CVAE (Conditional Variational Autoencoder) for butterfly image generation.
Encoder: Image + location → latent distribution (μ, σ)
Decoder: Latent sample + location → reconstructed image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """Convolutional block: Conv → BatchNorm → ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Transposed convolution block: ConvT → BatchNorm → ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, output_padding: int = 0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                         stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class CVAEEncoder(nn.Module):
    """
    CVAE Encoder: Image + location → (μ, σ)
    
    For 512×512 input:
    - Conv layers with stride 2 each: 512 → 256 → 128 → 64 → 32
    - Flatten and concatenate location embedding
    - Dense layers to μ and σ
    """
    
    def __init__(self, encoder_channels: Tuple[int, ...], location_dim: int, 
                 latent_dim: int):
        super().__init__()
        
        self.location_dim = location_dim
        self.latent_dim = latent_dim
        
        # Build convolutional layers
        conv_layers = []
        for i in range(len(encoder_channels) - 1):
            conv_layers.append(ConvBlock(encoder_channels[i], encoder_channels[i+1]))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after conv layers
        # After 4 conv blocks with stride 2: 512 → 256 → 128 → 64 → 32
        # Each conv reduces spatial dims by 2, so 512 / 2^5 = 16
        # (Actually 4 blocks: 512 / 2^4 = 32)
        self.spatial_size = 32
        self.flattened_size = encoder_channels[-1] * (self.spatial_size ** 2)
        
        # Dense layers: flattened + location → latent
        self.fc1 = nn.Linear(self.flattened_size + location_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_sigma = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, location_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor of shape (batch, 3, 512, 512)
            location_emb: Location embedding of shape (batch, location_dim)
        
        Returns:
            mu, sigma: Tensors of shape (batch, latent_dim)
        """
        # Convolutional pathway
        h = self.conv_layers(x)  # (batch, channels, 32, 32)
        h = h.view(h.size(0), -1)  # Flatten: (batch, flattened_size)
        
        # Concatenate location
        h = torch.cat([h, location_emb], dim=1)  # (batch, flattened_size + location_dim)
        
        # Dense layers
        h = self.relu(self.fc1(h))  # (batch, 512)
        
        # Split into μ and σ
        mu = self.fc_mu(h)  # (batch, latent_dim)
        log_sigma = self.fc_sigma(h)  # (batch, latent_dim)
        sigma = torch.exp(0.5 * log_sigma)  # Ensure σ > 0
        
        return mu, sigma


class CVAEDecoder(nn.Module):
    """
    CVAE Decoder: Latent sample + location → reconstructed image
    
    Takes latent z and location, upsamples to 512×512 image.
    """
    
    def __init__(self, latent_dim: int, location_dim: int, 
                 decoder_channels: Tuple[int, ...]):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.location_dim = location_dim
        
        # Initial spatial size (after deconv layers)
        self.spatial_size = 32
        initial_channels = decoder_channels[0]
        
        # Dense layers: latent + location → spatial feature map
        self.fc1 = nn.Linear(latent_dim + location_dim, 512)
        self.fc2 = nn.Linear(512, initial_channels * (self.spatial_size ** 2))
        self.relu = nn.ReLU(inplace=True)
        
        # Build deconvolutional layers
        deconv_layers = []
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1]
            # Last layer: special handling to reach 512×512
            if i == len(decoder_channels) - 2:
                # Final deconv to image: 256 → 512
                deconv_layers.append(DeconvBlock(in_ch, out_ch, output_padding=0))
            else:
                deconv_layers.append(DeconvBlock(in_ch, out_ch, output_padding=0))
        
        self.deconv_layers = nn.Sequential(*deconv_layers)
        
        # Output activation (Sigmoid to [0, 1] or can use [-1, 1] with Tanh)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor, location_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent sample of shape (batch, latent_dim)
            location_emb: Location embedding of shape (batch, location_dim)
        
        Returns:
            image: Reconstructed image of shape (batch, 3, 512, 512)
        """
        # Concatenate latent and location
        h = torch.cat([z, location_emb], dim=1)  # (batch, latent_dim + location_dim)
        
        # Dense layers to get spatial feature map
        h = self.relu(self.fc1(h))  # (batch, 512)
        h = self.relu(self.fc2(h))  # (batch, initial_channels * 32²)
        
        # Reshape to spatial feature map
        h = h.view(h.size(0), -1, self.spatial_size, self.spatial_size)
        # Shape: (batch, initial_channels, 32, 32)
        
        # Deconvolutional pathway
        x_recon = self.deconv_layers(h)  # (batch, 3, 512, 512)
        
        # Output activation
        x_recon = self.output_activation(x_recon)
        
        return x_recon


class CVAE(nn.Module):
    """
    Full CVAE model: combines encoder, decoder, and location encoding.
    """
    
    def __init__(self, encoder_channels: Tuple[int, ...], 
                 decoder_channels: Tuple[int, ...],
                 location_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        
        self.encoder = CVAEEncoder(encoder_channels, location_dim, latent_dim)
        self.decoder = CVAEDecoder(latent_dim, location_dim, decoder_channels)
        
        self.latent_dim = latent_dim
        self.location_dim = location_dim
    
    def encode(self, x: torch.Tensor, location_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode: image + location → (μ, σ)"""
        return self.encoder(x, location_emb)
    
    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: μ + σ * ε where ε ~ N(0, I)"""
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def decode(self, z: torch.Tensor, location_emb: torch.Tensor) -> torch.Tensor:
        """Decode: latent + location → image"""
        return self.decoder(z, location_emb)
    
    def forward(self, x: torch.Tensor, location_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass during training.
        
        Args:
            x: Image tensor (batch, 3, 512, 512)
            location_emb: Location embedding (batch, location_dim)
        
        Returns:
            x_recon: Reconstructed image
            mu: Mean of latent distribution
            sigma: Std dev of latent distribution
        """
        # Encode
        mu, sigma = self.encode(x, location_emb)
        
        # Reparameterize
        z = self.reparameterize(mu, sigma)
        
        # Decode
        x_recon = self.decode(z, location_emb)
        
        return x_recon, mu, sigma
    
    def sample(self, location_emb: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate new images for given locations by sampling from prior.
        
        Args:
            location_emb: Location embedding (batch, location_dim)
            num_samples: Number of samples per location (default 1)
        
        Returns:
            samples: Generated images (batch * num_samples, 3, 512, 512)
        """
        batch_size = location_emb.size(0)
        device = location_emb.device
        
        # Sample from standard normal
        z = torch.randn(batch_size * num_samples, self.latent_dim, device=device)
        
        # Repeat locations for each sample
        location_emb_repeated = location_emb.repeat_interleave(num_samples, dim=0)
        # (batch * num_samples, location_dim)
        
        # Decode
        samples = self.decode(z, location_emb_repeated)
        
        return samples


if __name__ == "__main__":
    # Test model
    from config import get_config
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CVAE(
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        location_dim=config.location_dim,
        latent_dim=config.latent_dim
    ).to(device)
    
    # Test forward pass
    x = torch.randn(4, 3, 512, 512).to(device)
    location_emb = torch.randn(4, config.location_dim).to(device)
    
    x_recon, mu, sigma = model(x, location_emb)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
