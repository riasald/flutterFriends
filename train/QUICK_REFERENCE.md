# Quick Reference: CVAE Training Commands

## Local Device (Data Preparation)

```bash
# 1. Download metadata (15-30 min)
cd scripts
python download_metadata.py

# 2. Download and filter images (30-45 min)
python download_filter_resumable.py

# 3. Verify dataset
wc -l ../data/filtered_metadata/metadata_quality.csv  # Should be 15k-25k
ls ../data/images_licensed/ | wc -l                   # Should match CSV rows
```

## Cloud GPU (Colab / Lambda Labs / Vast.ai)

### Setup (First Time)
```bash
# Install dependencies
pip install -r train/requirements.txt

# Verify installation
python train/verify_setup.py

# Copy verification output
# Should show all ✓ checks passing
```

### Training (Main Commands)
```bash
# Start training (choose one based on your setup)

# Option 1: Basic training
python train/train.py

# Option 2: With TensorBoard monitoring (separate terminal)
tensorboard --logdir logs/

# Option 3: With GPU memory tracking
nvidia-smi -l 1  # Update every 1 second (separate terminal)
```

### Monitoring Progress
```bash
# Check recent checkpoints
ls -lth models/

# View latest loss values (if TensorBoard not available)
tail -20 logs/*/events.out.tfevents*

# Expected epoch output:
# "Epoch 1/50: Train Loss=0.0450, KL=0.0100, β=0.1000"
```

## Configuration Tuning

### Common Adjustments (in `train/config.py`)

```python
# If running out of VRAM:
batch_size: int = 16  # Reduce from 32

# If training too slow:
batch_size: int = 64  # Increase from 32 (A100 only)

# If KL divergence stays 0:
beta_warmup_epochs: int = 20  # Increase from 10
beta_init: float = 0.01  # Increase from 0.0

# To train longer:
epochs: int = 100  # Increase from 50

# To use simpler (faster) location encoding:
location_encoding: str = "simple"  # Instead of "fourier"
```

## Post-Training (After ~6-10 hours)

### Download Checkpoint
```bash
# From local terminal (download checkpoint from cloud)
scp -i your_key username@cloud-instance:/path/to/models/cvae_butterfly_512x512_best.pt ./models/

# Or use cloud's file browser (Colab, Vast.ai, etc.)
```

### Generate Butterflies
```python
from train.inference import CVAEInference

# Load model
infer = CVAEInference("models/cvae_butterfly_512x512_best.pt")

# Single butterfly at location
image = infer.generate_image(lat=25.76, lon=-80.19)  # Miami

# Multiple variations (diverse at same location)
variations = infer.generate_variations(lat=40.71, lon=-74.01, num_samples=8)

# Interpolate between locations
interpolated = infer.interpolate_locations(
    lat1=25.76, lon1=-80.19,    # Miami
    lat2=34.05, lon2=-118.24,   # Los Angeles
    num_steps=10
)

# Visualize results
infer.visualize_variations(variations)
infer.visualize_interpolation(interpolated)
```

## Troubleshooting

### Problem: Out of Memory (CUDA out of memory)
```python
# Solution: In config.py, reduce batch size
batch_size: int = 8  # (was 32)
```

### Problem: Posterior Collapse (KL stays 0)
```python
# Solution: In config.py, adjust beta schedule
beta_warmup_epochs: int = 20  # (was 10, increase warmup)
beta_init: float = 0.01       # (was 0.0, start higher)
```

### Problem: Loss not decreasing
```python
# Solution: In config.py, try smaller learning rate
learning_rate: float = 5e-4  # (was 1e-3)
```

### Problem: "FileNotFoundError: metadata_quality.csv"
```bash
# Solution: Make sure you ran data download on local device first
# On your local machine:
cd scripts
python download_metadata.py
python download_filter_resumable.py
# Then transfer flutterFriends/ to cloud
```

### Problem: Slow training on GPU
```bash
# Check GPU utilization
nvidia-smi

# If util is low (<50%), try larger batch size
batch_size: int = 64  # (in config.py)

# If util is 100% but slow, GPU might be bottlenecked
# → Try different cloud provider or faster GPU
```

## File Locations

```
flutterFriends/
├── train/
│   ├── config.py         ← Edit for hyperparameters
│   ├── train.py          ← Run this to start training
│   ├── inference.py      ← Use this for generation
│   ├── models.py         ← CVAE architecture (don't edit)
│   ├── dataset.py        ← Data loading (don't edit)
│   └── verify_setup.py   ← Run to validate setup
│
├── data/
│   ├── filtered_metadata/
│   │   └── metadata_quality.csv   ← CSV with lat/lon/image_path
│   └── images_licensed/           ← Butterfly JPEGs
│
├── models/               ← Checkpoints saved here
│   └── cvae_butterfly_512x512_best.pt  ← Download after training
│
└── logs/                 ← TensorBoard logs
    └── events.out.tfevents*
```

## Expected Training Times

| GPU | Batch Size | Time/Epoch | Total (50 ep) | Cost |
|-----|-----------|-----------|---------------|------|
| A100 | 32 | 7 min | 6 hours | ~$15 |
| V100 | 32 | 12 min | 10 hours | ~$5 |
| Colab GPU | 16 | 15 min | 12 hours | Free/Pro |
| RTX 4060 | 4 | 3.5 hr | 175 hours | ❌ Too slow |

## Key Hyperparameters & Defaults

| Parameter | Value | Purpose |
|-----------|-------|---------|
| batch_size | 32 | Loss stability, speed (adjust for VRAM) |
| epochs | 50 | Full convergence with 20k images |
| learning_rate | 1e-3 | Adam optimizer |
| latent_dim | 128 | Butterfly representation capacity |
| beta_init | 0.0 | Start KL annealing at 0 |
| beta_max | 1.0 | End KL annealing at 1.0 |
| beta_warmup_epochs | 10 | Reach β_max over 10 epochs |
| fourier_freqs | 32 | Location encoding smoothness (32 → 128D) |
| encoder_channels | (3,16,32,64,64) | Progressive downsampling |

## Success Indicators

✅ **Good training looks like:**
- Loss: 0.040 → 0.025 (smooth decrease)
- KL divergence: 0 → 0.15-0.25 (increasing after warmup)
- β: 0.0 → 1.0 (rising over first 10 epochs)
- Validation loss follows training loss

❌ **Bad training looks like:**
- Loss flat or increasing
- KL stuck at 0 (posterior collapse)
- Validation loss diverging from training (overfitting)
- GPU utilization <50% (data bottleneck)

---

**Pro Tip**: Save this file to your cloud environment for quick reference during training!
