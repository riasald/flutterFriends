"""
Main training script for CVAE on butterfly images with location conditioning.
Implements β-annealing to prevent posterior collapse.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from config import get_config
from models import CVAE
from dataset import ButterflyDataLoader


class CVAETrainer:
    """Trainer for CVAE with β-annealing and checkpointing."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = CVAE(
            encoder_channels=config.encoder_channels,
            decoder_channels=config.decoder_channels,
            location_dim=config.location_dim,
            latent_dim=config.latent_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (optional: decay after plateau)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,  # Decay after 20 epochs
            gamma=0.8
        )
        
        # Loss function
        if config.reconstruction_loss == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction='mean')
        else:
            self.recon_loss_fn = nn.L1Loss(reduction='mean')
        
        # TensorBoard logger
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        
        print(f"Model initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between N(μ, σ²) and N(0, I).
        
        KL = -0.5 * mean(1 + log(σ²) - μ² - σ²)
        """
        return -0.5 * torch.mean(1.0 + torch.log(sigma**2 + 1e-8) - mu**2 - sigma**2)
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                    mu: torch.Tensor, sigma: torch.Tensor, 
                    beta: float) -> tuple:
        """
        CVAE loss: Reconstruction + β * KL divergence
        
        Args:
            x: Original images
            x_recon: Reconstructed images
            mu: Latent mean
            sigma: Latent std dev
            beta: KL weight (scheduled during training)
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss
        recon_loss = self.recon_loss_fn(x_recon, x)
        
        # KL divergence
        kl_loss = self.compute_kl_divergence(mu, sigma)
        
        # Total loss with β weighting
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_beta(self, epoch: int) -> float:
        """
        Beta schedule: linearly increase from beta_init to beta_max over warmup epochs.
        """
        if epoch < self.config.beta_warmup_epochs:
            # Linear warmup
            beta = self.config.beta_init + (self.config.beta_max - self.config.beta_init) * \
                   (epoch / self.config.beta_warmup_epochs)
        else:
            # Hold at max
            beta = self.config.beta_max
        return beta
    
    def train_epoch(self, train_loader, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        beta = self.get_beta(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for batch_idx, (images, locations) in enumerate(pbar):
            images = images.to(self.device)
            locations = locations.to(self.device)
            
            # Forward pass
            x_recon, mu, sigma = self.model(images, locations)
            
            # Compute loss
            loss, recon_loss, kl_loss = self.compute_loss(
                images, x_recon, mu, sigma, beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/recon_loss', recon_loss.item(), self.global_step)
                self.writer.add_scalar('train/kl_loss', kl_loss.item(), self.global_step)
                self.writer.add_scalar('train/beta', beta, self.global_step)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'beta': beta
            })
            
            self.global_step += 1
        
        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_kl = total_kl_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl,
            'beta': beta
        }
    
    def validate(self, val_loader, epoch: int) -> dict:
        """Validation pass."""
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        beta = self.get_beta(epoch)
        
        with torch.no_grad():
            for images, locations in val_loader:
                images = images.to(self.device)
                locations = locations.to(self.device)
                
                # Forward pass
                x_recon, mu, sigma = self.model(images, locations)
                
                # Compute loss
                loss, recon_loss, kl_loss = self.compute_loss(
                    images, x_recon, mu, sigma, beta
                )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Average metrics
        avg_loss = total_loss / len(val_loader)
        avg_recon = total_recon_loss / len(val_loader)
        avg_kl = total_kl_loss / len(val_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/recon_loss', avg_recon, epoch)
        self.writer.add_scalar('val/kl_loss', avg_kl, epoch)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"{self.config.checkpoint_name}_epoch_{epoch:03d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.output_dir,
                f"{self.config.checkpoint_name}_best.pt"
            )
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, train_loader, val_loader):
        """Full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.start_epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate step
            self.scheduler.step()
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}, "
                  f"β: {train_metrics['beta']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"KL: {val_metrics['kl_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            
            # Save best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
        
        self.writer.close()
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to {self.config.output_dir}")


def main():
    """Main training entry point."""
    config = get_config()
    
    print("\n" + "="*60)
    print("CVAE Training for Butterfly Location-Conditioned Generation")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}×{config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Location encoding: {config.location_encoding}")
    print(f"β schedule: {config.beta_init} → {config.beta_max} over {config.beta_warmup_epochs} epochs")
    print("="*60 + "\n")
    
    # Create trainer
    trainer = CVAETrainer(config)
    
    # Load data
    print("Loading dataset...")
    try:
        train_loader, val_loader = ButterflyDataLoader.get_train_val_loaders(
            metadata_csv=config.metadata_csv,
            images_dir=config.images_dir,
            batch_size=config.batch_size,
            train_split=config.train_split,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=config.seed,
            image_size=config.image_size,
            location_encoding=config.location_encoding,
            fourier_freqs=config.fourier_freqs
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease ensure:")
        print("1. Data has been downloaded (run download_metadata.py and download_filter_resumable.py)")
        print("2. Metadata CSV exists at:", config.metadata_csv)
        print("3. Images directory exists at:", config.images_dir)
        sys.exit(1)
    
    # Train
    print(f"Starting training with {len(train_loader)} training batches "
          f"and {len(val_loader)} validation batches...\n")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
