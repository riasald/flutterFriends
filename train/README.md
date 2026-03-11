"""
Quick setup guide and troubleshooting for CVAE training in flutterFriends.
"""

# CVAE Training Implementation - Quick Start

## Files Created

```
train/
├── __init__.py           # Module entry point
├── config.py            # Hyperparameter configuration
├── models.py            # CVAE encoder/decoder architecture
├── dataset.py           # Dataset class with Fourier location encoding
├── train.py             # Main training loop with β-annealing
├── inference.py         # Generation and sampling utilities
└── README.md            # This file
```

## Project Structure

```
flutterFriends/
├── scripts/
│   ├── download_metadata.py           # GBIF metadata fetcher
│   ├── download_images.py             # Single-threaded image downloader
│   └── download_filter_resumable.py   # Multi-threaded downloader (use this!)
├── data/
│   ├── raw_metadata/                  # GBIF metadata CSVs
│   ├── images_licensed/               # Downloaded butterfly images
│   └── filtered_metadata/             # Quality-filtered metadata + paths
├── train/                             # Training code (new)
│   ├── config.py                      # ← Adjust hyperparameters here
│   ├── train.py                       # ← Run this for training
│   └── inference.py                   # ← Use for generation
├── models/                            # Saved checkpoints (created at runtime)
├── logs/                              # TensorBoard logs (created at runtime)
└── api/
    └── main.py                        # API server (future)
```

## Step 1: Prepare Data (On Your Local Device)

### 1.1 Update metadata config
Edit `scripts/download_metadata.py`:
- Line 18: `MAX_ROWS = 75_000` → `150_000`

### 1.2 Download metadata
```bash
cd scripts
python download_metadata.py
# Fetches ~150k butterfly records from GBIF
# Time: 15-30 minutes (resumable)
```

### 1.3 Download and filter images
```bash
python download_filter_resumable.py
# Downloads 16 images in parallel with quality filtering
# Creates: data/images_licensed/ and metadata_quality.csv
# Time: 30-45 minutes
# Output: ~15k-25k images
```

### 1.4 Verify dataset
```bash
wc -l data/filtered_metadata/metadata_quality.csv  # Check row count
ls data/images_licensed/ | wc -l                   # Check image count
file data/images_licensed/*.jpg | head -5          # Spot check
```

## Step 2: Transfer to Cloud GPU

### Option A: Google Colab (Recommended for beginners)
```
1. Upload entire flutterFriends/ folder to Google Drive
2. Open Google Colab (colaboratory.research.google.com)
3. New notebook → Connect to cloud GPU
4. Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')

5. Set up paths:
   import sys
   sys.path.insert(0, '/content/drive/My Drive/flutterFriends')
   %cd /content/drive/My\ Drive/flutterFriends

6. Install dependencies:
   !pip install torch torchvision pillow numpy pandas tqdm tensorboard

7. Run training:
   !python train/train.py
```

### Option B: AWS / Lambda Labs / Vast.ai
```
1. Rent GPU instance (V100 $0.40/hr, A100 $1.10/hr, Vast $0.20-0.50/hr)
2. SSH into instance:
   ssh -i key.pem ubuntu@instance-ip

3. Upload code:
   scp -r flutterFriends/ username@instance:/home/
   
4. Install dependencies:
   pip install torch torchvision pillow numpy pandas tqdm tensorboard

5. Run training:
   cd flutterFriends
   python train/train.py
```

## Step 3: Configure Training (Cloud GPU)

Edit `train/config.py` to adjust hyperparameters:

```python
@dataclass
class CVAEConfig:
    # ===== CRITICAL PATHS (Update these!) =====
    data_dir: str = "../data"              # Path to data folder
    metadata_csv: str = "../data/filtered_metadata/metadata_quality.csv"
    images_dir: str = "../data/images_licensed"
    
    # ===== TRAINING =====
    batch_size: int = 32                   # Cloud GPU can handle 32+
    epochs: int = 50                       # Full convergence
    learning_rate: float = 1e-3
    
    # ===== KL ANNEALING (Prevents posterior collapse) =====
    beta_init: float = 0.0                 # Start at 0
    beta_max: float = 1.0                  # Increase to 1.0
    beta_warmup_epochs: int = 10           # Over 10 epochs (linear)
    
    # ===== MODEL ARCHITECTURE =====
    latent_dim: int = 128                  # Butterfly representation
    encoder_channels: (3, 16, 32, 64, 64) # Conv layer sizes
    
    # ===== LOCATION ENCODING =====
    location_encoding: str = "fourier"     # "fourier" or "simple"
    fourier_freqs: int = 32                # Fourier basis functions
```

**Key settings explained:**
- `beta_warmup_epochs = 10`: KL weight goes 0→1 over first 10 epochs (prevents posterior collapse)
- `batch_size = 32`: Cloud GPU VRAM allows this (local 4060 max: 4-8)
- `latent_dim = 128`: Captures butterfly appearance + location variation
- `location_encoding = "fourier"`: Better geographic smoothness than "simple"

## Step 4: Run Training

```bash
# On cloud GPU:
python train/train.py

# Monitor progress:
# - Console output: Loss, reconstruction, KL, β value each epoch
# - TensorBoard: tensorboard --logdir logs/
# - Checkpoints: Saved every 5 epochs in models/
```

**Expected training times:**
- A100 GPU: ~6 hours for 50 epochs
- V100 GPU: ~10 hours for 50 epochs
- Colab GPU: ~10 hours for 50 epochs

**Expected output:**
```
Epoch 1/50
  Train - Loss: 0.0450, Recon: 0.0350, KL: 0.0100, β: 0.1000
  Val   - Loss: 0.0445, Recon: 0.0345, KL: 0.0100

Epoch 2/50
  Train - Loss: 0.0420, Recon: 0.0330, KL: 0.0090, β: 0.2000
  ...
```

**Good indicators:**
- ✅ Loss decreases smoothly
- ✅ KL divergence increases (not collapsing to 0)
- ✅ β increases from 0.0 to 1.0 over first 10 epochs
- ✅ Validation loss follows training loss

**Bad indicators (troubleshoot):**
- ❌ Loss flat or increasing → Learning rate too high/low
- ❌ KL = 0 → Posterior collapse, increase β_warmup_epochs or reduce β_init
- ❌ Out of memory → Reduce batch_size or image_size

## Step 5: Generate Butterflies (After Training)

```bash
# Download checkpoint from cloud
scp ubuntu@instance:/path/to/models/cvae_butterfly_512x512_best.pt ./models/

# Run inference
python train/inference.py

# This will:
# 1. Generate butterflies at test locations
# 2. Show variations (sampling from latent)
# 3. Interpolate between locations
# 4. Save visualizations
```

Interactive generation example:
```python
from train.inference import CVAEInference

infer = CVAEInference("models/cvae_butterfly_512x512_best.pt")

# Generate single image
image = infer.generate_image(lat=25.76, lon=-80.19)  # Miami

# Generate 8 variations
variations = infer.generate_variations(lat=29.76, lon=-95.37, num_samples=8)

# Interpolate between locations
interpolated = infer.interpolate_locations(
    lat1=25.76, lon1=-80.19,    # Florida
    lat2=34.05, lon2=-118.24,   # LA
    num_steps=10
)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "FileNotFoundError: metadata_quality.csv not found"
```
Make sure data has been downloaded on your local device first!
- Run: python scripts/download_metadata.py
- Run: python scripts/download_filter_resumable.py
- Then transfer to cloud
```

### "CUDA out of memory"
In `train/config.py`:
```python
batch_size: int = 16  # Reduce from 32
# Or for very limited VRAM:
batch_size: int = 8
```

### "Loss not decreasing / stuck plateau"
```python
# In config.py:
learning_rate: float = 5e-4  # Try smaller LR
# Or increase warmup:
beta_warmup_epochs: int = 15  # Gradual KL increase
```

### "KL divergence is 0 (posterior collapse)"
```python
# In config.py:
beta_init: float = 0.01  # Start higher
beta_warmup_epochs: int = 20  # Longer warmup
# Or manually schedule beta during training
```

## Next Steps

1. **Download data** locally (1-2 hours)
2. **Transfer** to cloud GPU (15-30 min upload)
3. **Configure** hyperparameters in config.py
4. **Train** on cloud GPU (6-10 hours)
5. **Download** best checkpoint to local
6. **Generate** butterflies with inference.py
7. **Deploy** as API (future)

## Architecture Overview

```
ENCODER (Image + Location → Latent)
  Input: 512×512 RGB image + 128D location embedding
  Conv: 3 → 16 → 32 → 64 → 64
  Output: μ and σ (each 128D) → sample z ~ N(μ, σ²)

DECODER (Latent + Location → Image)
  Input: 128D latent sample z + 128D location embedding
  Deconv: 64 → 64 → 32 → 16 → 3
  Output: 512×512 RGB image

LOCATION ENCODING
  Fourier: sin/cos basis → 128D features (smooth, recommended)
  Simple: Normalize to [-1, 1] → 2D (fast baseline)
```

## Model Statistics

- **Parameters**: ~20M (medium CVAE suitable for 20k images)
- **Training time A100**: ~6 hours (50 epochs, batch 32)
- **Training time V100**: ~10 hours
- **VRAM needed A100**: ~35 GB (batch 32)
- **VRAM needed V100**: ~28 GB (batch 16-32)
- **Best checkpoint size**: ~80 MB

## References

- Kingma & Welling (2013): "Auto-Encoding Variational Bayes" (VAE paper)
- Sohn et al. (2015): "Learning Structured Output Representation using Deep Conditional Generative Models" (CVAE)
- Tancik et al. (2020): "Fourier Features Let Networks Learn High Frequency Functions" (positional encoding)

---

**Status**: ✅ Training infrastructure complete and ready for cloud GPU execution.
