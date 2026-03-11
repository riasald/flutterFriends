# 🚀 START HERE: Moving to New Device

**Read this first when you transfer to the new device.**

---

## What's Been Done ✅

The entire CVAE butterfly image generation project is complete and ready for continuation:

- ✅ **Core Implementation**: ~1,650 lines of production code (train/ folder)
- ✅ **Documentation**: Comprehensive guides for setup, architecture, design choices
- ✅ **Data Pipeline**: Existing scripts ready to download 20k-25k butterfly images
- ✅ **Infrastructure**: Ready for cloud GPU training (6-10 hours on A100/V100)

**Status**: Implementation complete, awaiting data download & cloud training

---

## Three Files You Need to Read (In Order)

### 1️⃣ Read This First: [TRANSFER_CHECKLIST.md](TRANSFER_CHECKLIST.md)
**Time**: 5 minutes
**Purpose**: Step-by-step guide for YOUR new device
**Contains**:
- Phase 1: Setup & verify (15 min)
- Phase 2: Download data (1-2 hours)
- Phase 3-8: Review, cloud setup, training, inference
- Troubleshooting
**Action**: Follow this document exactly

### 2️⃣ Understand the Project: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
**Time**: 20 minutes
**Purpose**: Understand what you're building and why
**Contains**:
- Complete project overview
- CVAE architecture explained
- Location encoding (Fourier features)
- Posterior collapse prevention (β-annealing)
- Data pipeline status
- Hardware strategy (6-10h on cloud vs 175h local)
**Action**: Read when you have context questions

### 3️⃣ Deep Dive (Optional): [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)
**Time**: 30 minutes
**Purpose**: Understand WHY each design choice was made
**Contains**:
- CVAE vs GANs vs Diffusion comparison
- Fourier encoding mathematical justification
- β-annealing: why it prevents collapse
- Architecture dimensions: 3→16→32→64→64 reasoning
- Learning rate: why 1e-3 (alternative analysis)
- All trade-offs explained
**Action**: Read if you want to modify hyperparameters or understand decisions deeply

---

## Quick Start on New Device

### If You Have 30 Minutes:
```bash
# 1. Read TRANSFER_CHECKLIST.md
cat TRANSFER_CHECKLIST.md

# 2. Do Phase 1: Setup & verification (15 min)
cd train
python verify_setup.py
```

### If You Have 2+ Hours:
```bash
# Follow TRANSFER_CHECKLIST.md Phase 2: Data acquisition
cd scripts
python download_metadata.py        # 30 min
python download_filter_resumable.py  # 45 min

# Then transfer to cloud GPU for training
```

### If You Have 12+ Hours (Full Training):
```bash
# Complete phases 1-6 from TRANSFER_CHECKLIST.md
# Results in trained model + generated butterflies
```

---

## File Organization

```
✅ TRANSFER DOCUMENTATION (Read these first)
├── TRANSFER_CHECKLIST.md       ← Step-by-step for your device
├── PROJECT_CONTEXT.md          ← What, why, architecture
├── DESIGN_DECISIONS.md         ← Why each choice (optional deep dive)
└── TODO.md                     ← Project status

✅ CORE CODE (Already implemented)
train/
├── config.py                   ← Hyperparameters (don't modify unless needed)
├── models.py                   ← CVAE architecture
├── dataset.py                  ← Data loading
├── train.py                    ← Run this to train
├── inference.py                ← Run this to generate butterflies
└── [other supporting files]

✅ DATA PIPELINE (Use these to download data)
scripts/
├── download_metadata.py        ← Fetch metadata (~150k records)
└── download_filter_resumable.py ← Download & filter images (~20k-25k)

✅ DATA (After download)
data/
├── filtered_metadata/metadata_quality.csv  ← CSV with lat, lon, image_path
└── images_licensed/                        ← 15k-25k butterfly JPEGs
```

---

## Your Immediate Action Items

### Step 1: Verify Project Transferred ✓ (5 min)
```bash
# Confirm all files exist
cd flutterFriends
ls TRANSFER_CHECKLIST.md PROJECT_CONTEXT.md DESIGN_DECISIONS.md train/ scripts/
# Should show all files
```

### Step 2: Read TRANSFER_CHECKLIST.md ✓ (5 min)
```bash
cat TRANSFER_CHECKLIST.md
# Or open in your editor
```

### Step 3: Follow Phase 1 (15 min)
From TRANSFER_CHECKLIST.md Phase 1:
```bash
cd train
pip install -r requirements.txt
python verify_setup.py
```

### Step 4: Download Data (1-2 hours)
From TRANSFER_CHECKLIST.md Phase 2:
```bash
cd scripts
python download_metadata.py
python download_filter_resumable.py
```

### Step 5: Move to Cloud GPU
From TRANSFER_CHECKLIST.md Phase 4:
- Choose: Google Colab (easiest) or Vast.ai (budget)
- Upload flutterFriends/ folder
- Follow cloud-specific setup

### Step 6: Train (6-10 hours)
From TRANSFER_CHECKLIST.md Phase 5:
```bash
python train/train.py
# Monitor with: tensorboard --logdir logs/
```

### Step 7: Generate Butterflies (Post-training)
From TRANSFER_CHECKLIST.md Phase 6:
```bash
python train/inference.py
# Or use interactive generation
```

---

## Key Information At-A-Glance

| Question | Answer | Reference |
|----------|--------|-----------|
| What am I building? | CVAE to generate butterfly images by location | PROJECT_CONTEXT.md |
| Why CVAE? | Conditioning + diversity + stability | DESIGN_DECISIONS.md |
| Why Fourier encoding? | Smooth geographic space, enables interpolation | DESIGN_DECISIONS.md |
| What about posterior collapse? | β-annealing prevents it | DESIGN_DECISIONS.md |
| How long to train? | 6h (A100) vs 175h (RTX 4060) → use cloud | PROJECT_CONTEXT.md |
| Where do I start? | Follow TRANSFER_CHECKLIST.md Phase by Phase | TRANSFER_CHECKLIST.md |
| What if something fails? | Check troubleshooting in TRANSFER_CHECKLIST.md | TRANSFER_CHECKLIST.md |
| Can I change hyperparameters? | Yes, edit train/config.py before training | train/README.md |
| Do I need GPU? | Not for setup/data download, YES for training | TRANSFER_CHECKLIST.md |

---

## Expected Timeline

```
New Device Setup:        15 min    (Phase 1)
Data Download:           1-2 hrs   (Phase 2)
Cloud Transfer:          30 min    (Phase 3)
Cloud Setup:             5 min     (Phase 4)
Training:                6-10 hrs  (Phase 5, mostly waiting)
Inference:               5 min     (Phase 6)
────────────────────────────────
TOTAL:                   8-14 hrs (mostly waiting for GPU)
```

---

## Support Resources in Project

| Problem | Resource |
|---------|----------|
| How do I set up? | TRANSFER_CHECKLIST.md Phase 1 |
| How do I download data? | TRANSFER_CHECKLIST.md Phase 2 |
| GPU out of memory? | TRANSFER_CHECKLIST.md Phase 7 or train/README.md |
| KL divergence stuck at 0? | train/README.md or TRANSFER_CHECKLIST.md |
| Loss not decreasing? | train/QUICK_REFERENCE.md |
| How do I generate images? | TRANSFER_CHECKLIST.md Phase 6 |
| Why these design choices? | DESIGN_DECISIONS.md |
| What's the architecture? | PROJECT_CONTEXT.md |
| Quick commands? | train/QUICK_REFERENCE.md |

---

## One-Command Verification

After transfer, verify everything is ready:

```bash
# Check project files
ls -la | grep -E "TRANSFER|PROJECT|DESIGN|TODO"
# Should show: TRANSFER_CHECKLIST.md, PROJECT_CONTEXT.md, DESIGN_DECISIONS.md, TODO.md

# Check code files
ls train/*.py
# Should show: config.py, models.py, dataset.py, train.py, inference.py, etc.

# Check data scripts
ls scripts/download*.py
# Should show: download_metadata.py, download_filter_resumable.py, etc.
```

---

## Decision Tree: What To Do Next

```
Have 30 min?
  ├→ YES: Read TRANSFER_CHECKLIST.md + run verify_setup.py
  └→ NO: Come back later

Have 2 hours?
  ├→ YES: Run data download (Phase 2)
  └→ NO: Just verify setup now

Have access to GPU?
  ├→ Google Colab: YES → Follow Phase 4 (Colab setup)
  ├→ Local RTX 4060: YES → Skip local testing (use cloud instead)
  └→ Cloud provider: YES → Rent A100 (6h, $15) or V100 (10h, $5)

Ready to train?
  ├→ YES: Follow TRANSFER_CHECKLIST.md Phase 5
  ├→ Modify hyperparameters first: Edit train/config.py
  └→ Not ready: Check setup with python train/verify_setup.py
```

---

## Success Looks Like

### Phase 1 (Setup)
```
✓ verify_setup.py shows all green checks
✓ No import errors
✓ Can list demo image file
```

### Phase 2 (Data)
```
✓ metadata_quality.csv exists with 15k-25k rows
✓ images_licensed/ has same count of JPEGs
✓ Total size 30-50 GB
```

### Phase 6 (Training)
```
✓ TensorBoard shows loss decreasing
✓ KL divergence increases from 0 to 0.15-0.25
✓ β schedule visible: 0 → 1.0 over 10 epochs
✓ Checkpoints save every 5 epochs
```

### Phase 7 (Inference)
```
✓ Generated images look like real butterflies
✓ Different locations have different butterfly styles
✓ Sampling produces diversity (not identical)
```

---

## Remember

1. **Follow TRANSFER_CHECKLIST.md** step by step - it's written for YOUR situation
2. **All design choices are documented** in PROJECT_CONTEXT.md and DESIGN_DECISIONS.md
3. **Everything is ready to run** - just follow the phases
4. **Data download is the first blocker** - do this when you have 2 hours
5. **Cloud GPU is where you train** - don't use local RTX 4060 for main training
6. **Questions?** Check the reference table above

---

## Next: Open TRANSFER_CHECKLIST.md

This file guides you through every step on your new device.

```bash
cat TRANSFER_CHECKLIST.md
# Or open in your editor of choice
```

**Good luck! 🦋**
