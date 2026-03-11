# Transfer Checklist: Getting Started on New Device

**Purpose**: Step-by-step checklist to seamlessly continue the project on a new device

**Start here when you've transferred all project files to the new device.**

---

## Phase 1: Setup & Verification (15 minutes)

### 1.1 Verify Project Structure
```bash
# Navigate to project root
cd flutterFriends

# Verify all essential folders exist
ls -la  # Should see:
```
- ✅ `train/` folder (with .py files)
- ✅ `scripts/` folder (download utilities)
- ✅ `data/` folder (metadata + images if transferred)
- ✅ Documentation files (README.md, PROJECT_CONTEXT.md, etc.)

**If any folder missing**: Recopy the project from source device

### 1.2 Verify Data Transfer (if bringing images)
```bash
# Check if metadata CSV exists
ls data/filtered_metadata/metadata_quality.csv

# Count rows (should be 15k-25k)
wc -l data/filtered_metadata/metadata_quality.csv

# Check if images downloaded
ls data/images_licensed/ | wc -l  # Should be same number as CSV

# Check total size
du -sh data/images_licensed/  # Should be 30-50 GB
```

**If data NOT transferred**: You'll re-download in Phase 2

### 1.3 Create Virtual Environment (Recommended)
```bash
# Python 3.9 or 3.10 recommended
python --version  # Check version

# Create venv
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Should see (venv) prompt
```

### 1.4 Install Dependencies
```bash
# Navigate to train folder
cd train

# Install from requirements
pip install -r requirements.txt

# This installs:
# - torch, torchvision (deep learning)
# - pillow, numpy, pandas (data handling)
# - tqdm (progress bars)
# - tensorboard (monitoring)
# - matplotlib (visualization)
#
# Time: 5-15 minutes (depends on internet)
```

**Troubleshoot if pip stalls**:
```bash
# Use newer PyTorch index
pip install -r requirements.txt --upgrade

# Or install individually
pip install torch torchvision==0.16.0
pip install pillow numpy pandas
pip install tqdm tensorboard matplotlib
```

### 1.5 Verify Installation
```bash
# Run verification script
python verify_setup.py

# Should output:
# ✓ torch installed
# ✓ torchvision installed
# ✓ config.py loads successfully
# ✓ models.py loads successfully
# ✓ dataset.py loads successfully
# ✓ train.py loads successfully
# ✓ inference.py loads successfully
# ✓ ALL CHECKS PASSED!
```

**If any checks fail**: See `train/README.md` Troubleshooting section

---

## Phase 2: Data Acquisition (1-2 hours)

**Only do this if you did NOT transfer the data folder.**

### 2.1 Download Metadata from GBIF
```bash
cd scripts
python download_metadata.py

# What it does:
# - Fetches GBIF iNaturalist observations
# - Filter: Nymphalidae family, US, 2020-2024
# - Creates: ../data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv
#
# Time: 15-30 minutes
# Output: ~150k metadata rows

# Monitor progress:
wc -l ../data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv
# Should grow to 150000 rows
```

**Expected output last few lines**:
```
Fetching: offset=145000...
Downloaded 150000 records
Total records: 150,000
Saved to: ../data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv
```

### 2.2 Download & Filter Images
```bash
python download_filter_resumable.py

# What it does:
# - Downloads butterfly images (16 parallel threads)
# - Filters: CC0, CC-BY licenses only
# - Quality check: min 256×256 pixels
# - Creates: ../data/images_licensed/ (JPG files)
# - Creates: ../data/filtered_metadata/metadata_quality.csv
#
# Time: 30-45 minutes
# Output: 15k-25k images

# Monitor progress:
# Watch console output for "Downloaded: X/Y images"
# Or check folder size:
du -sh ../data/images_licensed/
# Should grow to 30-50 GB
```

**If interrupted**: Script is resumable. Just run again:
```bash
python download_filter_resumable.py
# Script checks what's already downloaded
# Skips completed downloads
# Resumes from checkpoint
```

### 2.3 Verify Dataset
```bash
# After downloading complete:

# 1. Check CSV row count
wc -l ../data/filtered_metadata/metadata_quality.csv
# Expected: 15000-25000 (rows should match images)

# 2. Check image count
ls ../data/images_licensed/ | wc -l
# Expected: 15000-25000 (should match CSV)

# 3. Spot check image files
file ../data/images_licensed/*.jpg | head -5
# Should show "JPEG image data" for all

# 4. Check total data size
du -sh ../data/
# Expected: 30-50 GB
```

**If counts don't match**: Some images may have failed. That's OK - continue anyway.

---

## Phase 3: Review Documentation (10 minutes)

### 3.1 Read Quick Reference
```bash
# Review command cheat sheet
cat train/QUICK_REFERENCE.md

# Key sections:
# - Common configuration adjustments
# - Cloud GPU setup
# - Training commands
# - Monitoring progress
# - Troubleshooting
```

### 3.2 Review Architecture
```bash
# Understand what's being built
cat train/models.py | head -100  # Architecture overview

# Check config settings
cat train/config.py | grep "^    [a-z]" | head -20
# Shows all hyperparameters
```

### 3.3 Read Context Documents
```bash
# Understand why design choices were made
# (Do this now if planning to modify hyperparameters)

# Recommended reading order:
# 1. PROJECT_CONTEXT.md (what, why, architecture)
# 2. DESIGN_DECISIONS.md (detailed rationale for each choice)
# 3. train/README.md (setup and troubleshooting)
```

---

## Phase 4: Prepare for Cloud Training (if applicable)

**Only do this if you have a GPU and want to train. Otherwise skip to Phase 5.**

### 4.1 Choose Cloud GPU Provider

| Provider | Setup Time | Cost | Quality | Recommendation |
|----------|-----------|------|---------|---|
| **Google Colab** | 5 min | Free/Pro | Good | ⭐ Beginner-friendly |
| **Vast.ai** | 10 min | $0.20-0.50/hr | Good | ⭐ Budget option |
| **Lambda Labs** | 10 min | $0.40-1.00/hr | Excellent | Good |
| **AWS** | 20 min | Variable | Good | Advanced users |

### 4.2 Google Colab Setup (Easiest)
```
1. Upload project to Google Drive
   - Right-click in Drive → New Folder → flutterFriends
   - Upload entire folder (30-50 GB if data included)
   - Or just scripts/ + train/ (small)

2. Open Colab
   - Go to colab.research.google.com
   - New notebook

3. Mount drive
   from google.colab import drive
   drive.mount('/content/drive')

4. Install dependencies
   !pip install -r /content/drive/flutterFriends/train/requirements.txt

5. Run training
   %cd /content/drive/flutterFriends
   !python train/train.py
```

### 4.3 Vast.ai Setup (Budget)
```
1. Create account: vast.ai
2. Rent GPU instance (A100 or V100)
3. SSH into instance
4. Upload project via SCP
5. Install dependencies
6. Run training
```

### 4.4 Configure Hyperparameters (Optional)
```bash
# Before training, you CAN adjust settings
# But defaults are well-tuned, so not necessary

cd train

# To see current settings:
grep -E "batch_size|epochs|learning_rate|latent_dim" config.py

# To modify (optional):
# Edit config.py with your editor
# Common adjustments:
#   batch_size: 32 (try 16-64 on different GPUs)
#   epochs: 50 (try 100 for more training)
#   beta_warmup_epochs: 10 (try 15-20 if KL divergence stuck at 0)
```

---

## Phase 5: Run Training

### 5.1 Start Training (Main Command)
```bash
# Ensure you're in project root
cd flutterFriends

# Run training
python train/train.py

# Expected output:
# Training CVAE on butterfly images
# Device: cuda (or cpu)
# Epoch 1/50: Loss=0.0450, Recon=0.0350, KL=0.0100, β=0.1000
# Epoch 2/50: Loss=0.0420, Recon=0.0330, KL=0.0090, β=0.2000
# ...
```

**Note**: First epoch will be slower (data loading, GPU warmup)

### 5.2 Monitor Progress (In Separate Terminal)
```bash
# Open another terminal in project root
tensorboard --logdir logs/

# Then open browser:
http://localhost:6006

# Graphs to watch:
# - train/loss (should decrease smoothly)
# - train/kl_divergence (should increase in first 10 epochs)
# - train/beta (should rise 0 → 1.0)
# - val/loss (should track train loss)
```

### 5.3 Expected Training Times
```
GPU          | Epochs | Time
A100         | 50     | ~6 hours
V100         | 50     | ~10 hours
Colab GPU    | 50     | ~12 hours
RTX 4090     | 50     | ~4 hours
RTX 3090     | 50     | ~8 hours
RTX 4060     | 50     | ~175 hours ❌ DON'T USE
```

### 5.4 Training Checkpoints
```bash
# During training, checkpoints save to:
ls models/

# Should see files like:
# checkpoint_epoch_5.pt
# checkpoint_epoch_10.pt
# ...
# cvae_butterfly_512x512_best.pt  (best validation loss)

# If training interrupted:
# Next run of train.py automatically resumes from latest checkpoint
```

---

## Phase 6: Generate Butterflies (After Training)

### 6.1 Check Training Completed
```bash
# Check if best model saved
ls models/cvae_butterfly_512x512_best.pt

# Check training logs
tensorboard --logdir logs/  # See final curves
# Or check last epoch output in console
```

### 6.2 Run Inference
```bash
# Option 1: Automatic inference (generates test images)
python train/inference.py
# Creates visualizations in project root

# Option 2: Interactive generation (Python)
python

# Inside Python:
from train.inference import CVAEInference
import torch

# Load model
infer = CVAEInference("models/cvae_butterfly_512x512_best.pt")

# Generate butterfly at specific location
image = infer.generate_image(lat=25.76, lon=-80.19)  # Miami
print("Generated image shape:", image.shape)

# Generate variations (diverse at same location)
variations = infer.generate_variations(lat=40.71, lon=-74.01, num_samples=5)
print("Generated", len(variations), "variations")

# Interpolate between two locations
path = infer.interpolate_locations(
    lat1=37.78, lon1=-122.41,   # San Francisco
    lat2=34.05, lon2=-118.24,   # Los Angeles
    num_steps=10
)
print("Interpolated path:", len(path), "images")

# Visualize
infer.visualize_variations(variations)
infer.visualize_interpolation(path)
```

---

## Phase 7: Troubleshooting

### Problem: "CUDA out of memory"
```python
# Solution: Reduce batch size in config.py
# Before:
batch_size: int = 32

# After:
batch_size: int = 16  # (try this)
# or
batch_size: int = 8   # (if still OOM)

# Then restart training
```

### Problem: "FileNotFoundError: metadata_quality.csv"
```bash
# Solution: Make sure you completed Phase 2
cd scripts
python download_metadata.py
python download_filter_resumable.py

# Then verify:
ls ../data/filtered_metadata/metadata_quality.csv
```

### Problem: "KL divergence stays at 0"
```python
# Solution: Increase β-annealing warmup in config.py
# Before:
beta_warmup_epochs: int = 10

# After:
beta_warmup_epochs: int = 20  # Give more time
# or
beta_init: float = 0.01  # Start KL penalty earlier

# Restart training
```

### Problem: "Loss is not decreasing"
```python
# Solution: Try smaller learning rate in config.py
# Before:
learning_rate: float = 1e-3

# After:
learning_rate: float = 5e-4  # Smaller LR
# or check your data isn't corrupted
```

### Problem: "Training is very slow"
```bash
# Check GPU utilization
nvidia-smi  # Should show >80% GPU

# If low GPU util:
# - Try larger batch_size
# - Check if data loading is bottleneck
# - May need faster GPU provider

# Check CPU usage:
top  # (or Task Manager on Windows)
# If one core pinned at 100%, data loading is slow
```

---

## Phase 8: Quick Reference Links

**During training or afterwards, you may need**:

```bash
# Architecture details
cat train/models.py | head -150

# Configure hyperparameters
nano train/config.py
# or: vim train/config.py
# or: open with your editor of choice

# View loss curves
tensorboard --logdir logs/

# Generate butterflies
python train/inference.py

# Check for errors
python train/verify_setup.py

# See detailed setup guide
cat train/README.md

# Command cheat sheet
cat train/QUICK_REFERENCE.md

# Project context
cat PROJECT_CONTEXT.md

# Why design choices
cat DESIGN_DECISIONS.md
```

---

## Success Checklist

### After Setup (Phase 1)
- [ ] Project folders verified
- [ ] Python virtual environment created
- [ ] Dependencies installed (`pip install -r train/requirements.txt`)
- [ ] verify_setup.py passes all checks

### After Data Prep (Phase 2)
- [ ] Metadata CSV exists: `data/filtered_metadata/metadata_quality.csv`
- [ ] Image count matches: `ls data/images_licensed/ | wc -l` ≈ rows in CSV
- [ ] Total size reasonable: 30-50 GB

### During Training (Phase 5)
- [ ] Console shows loss decreasing
- [ ] TensorBoard shows smooth curves
- [ ] GPU utilization >80% (check with nvidia-smi)
- [ ] Checkpoints saving to models/ folder
- [ ] KL divergence increases during first 10 epochs

### After Training (Phase 6)
- [ ] Best model saved: `models/cvae_butterfly_512x512_best.pt`
- [ ] Inference script runs without errors
- [ ] Generated images appear realistic

---

## When Stuck

**First step**: Check relevant README
- `train/README.md` → Detailed setup guide + troubleshooting
- `train/QUICK_REFERENCE.md` → Quick command tips
- `PROJECT_CONTEXT.md` → Architecture and decisions
- `DESIGN_DECISIONS.md` → Why each choice was made

**Then**: Run verification
```bash
python train/verify_setup.py
# See which check(s) failed
```

**Get diagnostic info**:
```bash
python --version
pip list | grep torch
nvidia-smi  # (if GPU available)
df -h       # (check disk space)
```

---

**You've got this! The project is well-documented and ready to run. Start with Phase 1, proceed sequentially, and reference the documentation as needed.**
