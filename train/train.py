"""
Main training script for CVAE on butterfly images with location conditioning.
Implements β-annealing to prevent posterior collapse.
Includes robust checkpointing, recovery, and logging for cloud training.
"""

import os
import sys
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time

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
        """Train one epoch with aggressive checkpoint saving."""
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
                
                # Log metrics to CSV
                self.log_training_metrics(epoch, batch_idx, {
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'beta': beta,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'beta': beta
            })
            
            # Save recovery checkpoint every 50 steps (for resuming if interrupted)
            if batch_idx % 50 == 0 and batch_idx > 0:
                self.save_checkpoint(epoch, step=batch_idx, is_recovery=True)
            
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
    
    def save_checkpoint(self, epoch: int, step: int = None, is_best: bool = False, is_recovery: bool = False):
        """Save model checkpoint with recovery capability."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'timestamp': time.time(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
        }
        
        # Save recovery checkpoint (latest, for resuming)
        if is_recovery:
            recovery_path = os.path.join(
                self.config.output_dir,
                "recovery_latest.pt"
            )
            torch.save(checkpoint, recovery_path)
            return
        
        # Save regular checkpoint with step info
        if step is not None:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                f"ckpt_ep{epoch:03d}_step{step:06d}.pt"
            )
        else:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                f"ckpt_ep{epoch:03d}.pt"
            )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.output_dir,
                f"{self.config.checkpoint_name}_best.pt"
            )
            torch.save(checkpoint, best_path)
        
        print(f"[Checkpoint] Saved: {Path(checkpoint_path).name}")
        
        # Clean up old checkpoints (keep only 3 latest)
        self._cleanup_old_checkpoints(keep=3)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        print(f"[Recovery] Loaded checkpoint: {Path(checkpoint_path).name}")
        print(f"           Resuming from epoch {self.start_epoch + 1}")
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints, keeping only the latest N."""
        checkpoint_dir = Path(self.config.output_dir)
        ckpts = sorted(checkpoint_dir.glob('ckpt_ep*.pt'))
        
        if len(ckpts) > keep:
            for old_ckpt in ckpts[:-keep]:
                old_ckpt.unlink()
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint for recovery."""
        checkpoint_dir = Path(self.config.output_dir)
        
        # First try recovery checkpoint (most recent)
        recovery_ckpt = checkpoint_dir / "recovery_latest.pt"
        if recovery_ckpt.exists():
            return str(recovery_ckpt)
        
        # Fall back to latest epoch checkpoint
        ckpts = sorted(checkpoint_dir.glob('ckpt_ep*.pt'))
        if ckpts:
            return str(ckpts[-1])
        
        return None
    
    def log_training_metrics(self, epoch: int, step: int, metrics: dict):
        """Log training metrics to CSV for analysis."""
        log_path = Path(self.config.output_dir) / "training_log.csv"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'total_steps': self.global_step,
            **metrics
        }
        
        # Write header if file doesn't exist
        if not log_path.exists():
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
        
        # Append metrics
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)
    
    def train(self, train_loader, val_loader):
        """Full training loop with recovery capability."""
        best_val_loss = float('inf')
        
        # Check for recovery checkpoint
        recovery_ckpt = self.find_latest_checkpoint()
        if recovery_ckpt:
            print(f"\n[Recovery] Found checkpoint: {Path(recovery_ckpt).name}")
            response = input("Resume training from checkpoint? (y/n) ")
            if response.lower() == 'y':
                self.load_checkpoint(recovery_ckpt)
        
        for epoch in range(self.start_epoch, self.config.epochs):
            try:
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validate
                val_metrics = self.validate(val_loader, epoch)
                
                # Learning rate step
                self.scheduler.step()
                
                # Print summary
                print(f"\n[Epoch {epoch+1}/{self.config.epochs}]")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Recon: {train_metrics['recon_loss']:.4f}, "
                      f"KL: {train_metrics['kl_loss']:.4f}, "
                      f"β: {train_metrics['beta']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Recon: {val_metrics['recon_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f}")
                
                # Save checkpoint every epoch
                self.save_checkpoint(epoch)
                
                # Save best
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, is_best=True)
            
            except KeyboardInterrupt:
                print("\n[Interrupted] Saving recovery checkpoint...")
                self.save_checkpoint(epoch, is_recovery=True)
                print("Training paused. Run again to resume from checkpoint.")
                return
            except Exception as e:
                print(f"\n[Error] {e}")
                print("Saving recovery checkpoint...")
                self.save_checkpoint(epoch, is_recovery=True)
                raise
        
        self.writer.close()
        print(f"\n[Complete] Training finished. Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to {self.config.output_dir}")


def main():
    """Main training entry point with recovery capability."""
    config = get_config()
    
    print("\n" + "="*60)
    print("CVAE Training for Butterfly Location-Conditioned Generation")
    print("="*60)
    print(f"Device: {config.device if torch.cuda.is_available() else 'CPU'}")
    print(f"Image size: {config.image_size}×{config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Location encoding: {config.location_encoding}")
    print(f"β schedule: {config.beta_init} → {config.beta_max} over {config.beta_warmup_epochs} epochs")
    print("="*60 + "\n")
    
    # Create trainer
    trainer = CVAETrainer(config)
    
    # Load data
    print("[1/3] Loading dataset...")
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
        print(f"✓ Loaded {len(train_loader)} training batches, {len(val_loader)} validation batches\n")
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        print("\nPlease ensure:")
        print("1. Data has been downloaded")
        print("2. Metadata CSV exists at:", config.metadata_csv)
        print("3. Images directory exists at:", config.images_dir)
        sys.exit(1)
    
    # Train
    print(f"[2/3] Starting training...")
    print(f"      Output directory: {config.output_dir}\n")
    try:
        trainer.train(train_loader, val_loader)
        print(f"\n[3/3] Generating sample images from trained model...")
        generate_samples_from_checkpoint(
            checkpoint_path=Path(config.output_dir) / f"{config.checkpoint_name}_best.pt",
            config=config,
            num_samples=10
        )
    except Exception as e:
        print(f"\n✗ Training interrupted: {e}")
        print("\nAttempting to generate samples from latest checkpoint...")
        recovery_ckpt = trainer.find_latest_checkpoint()
        if recovery_ckpt:
            try:
                generate_samples_from_checkpoint(recovery_ckpt, config, num_samples=10)
            except Exception as gen_err:
                print(f"Could not generate samples: {gen_err}")


def generate_samples_from_checkpoint(checkpoint_path, config, num_samples: int = 10):
    """Generate sample images from any checkpoint (useful if training interrupted)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    model = CVAE(
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        location_dim=config.location_dim,
        latent_dim=config.latent_dim
    ).to(device).eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    from dataset import fourier_encode_location
    from torchvision.utils import save_image
    
    sample_dir = Path(config.output_dir) / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    locations = [
        (40.7128, -74.0060, "NYC"),
        (34.0522, -118.2437, "LA"),
        (51.5074, -0.1278, "London"),
    ]
    
    with torch.no_grad():
        for lat, lon, name in locations:
            loc_emb = fourier_encode_location(lat, lon, config.fourier_freqs)
            loc_emb = torch.from_numpy(loc_emb).float().to(device).unsqueeze(0)
            
            for i in range(num_samples // len(locations)):
                z = torch.randn(1, config.latent_dim, device=device)
                img = model.decode(z, loc_emb)
                save_path = sample_dir / f"{name}_sample_{i}.jpg"
                save_image(img, save_path)
    
    print(f"✓ Generated samples in: {sample_dir}")


if __name__ == "__main__":
    main()
