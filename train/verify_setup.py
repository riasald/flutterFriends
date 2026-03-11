#!/usr/bin/env python3
"""
Validation script for CVAE training setup.
Run this to verify all modules are working correctly before training.

Usage:
    python verify_setup.py
"""

import sys
import importlib.util
from pathlib import Path

def check_module(module_name, package_name=None):
    """Check if a module is installed."""
    check_name = package_name or module_name
    try:
        __import__(check_name)
        print(f"✓ {module_name:20s} installed")
        return True
    except ImportError:
        print(f"✗ {module_name:20s} NOT FOUND - install with: pip install {package_name or module_name}")
        return False

def check_local_module(name, path):
    """Check if a local Python module is present and importable."""
    if not path.exists():
        print(f"✗ {name:20s} NOT FOUND at {path}")
        return False
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✓ {name:20s} loads successfully")
        return True
    except Exception as e:
        print(f"✗ {name:20s} ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("CVAE Training Setup Verification")
    print("=" * 60)
    
    # Check external dependencies
    print("\n[1/3] Checking external dependencies...")
    deps_ok = all([
        check_module("torch", "torch"),
        check_module("torchvision", "torchvision"),
        check_module("PIL", "pillow"),
        check_module("numpy"),
        check_module("torch.utils.tensorboard", "tensorboard"),
        check_module("tqdm"),
    ])
    
    if not deps_ok:
        print("\n❌ Missing dependencies. Install with:")
        print("   pip install torch torchvision pillow numpy tqdm tensorboard")
        return False
    
    # Check local modules
    print("\n[2/3] Checking local training modules...")
    train_dir = Path(__file__).parent
    
    local_modules = [
        ("config", train_dir / "config.py"),
        ("models", train_dir / "models.py"),
        ("dataset", train_dir / "dataset.py"),
        ("train", train_dir / "train.py"),
        ("inference", train_dir / "inference.py"),
    ]
    
    modules_ok = all(check_local_module(name, path) for name, path in local_modules)
    
    if not modules_ok:
        print("\n❌ Some modules are missing or broken.")
        return False
    
    # Check data directory structure
    print("\n[3/3] Checking data directories...")
    data_dir = train_dir.parent / "data"
    
    checks = [
        ("data/", data_dir),
        ("data/filtered_metadata/", data_dir / "filtered_metadata"),
        ("data/images_licensed/", data_dir / "images_licensed"),
    ]
    
    data_ok = True
    for name, path in checks:
        if path.exists():
            print(f"✓ {name:40s} exists")
            if path.name == "images_licensed":
                count = len(list(path.glob("*.jpg")))
                print(f"  └─ Contains {count} images")
        else:
            print(f"⚠ {name:40s} not found (will be created after data download)")
    
    # Try importing the train module
    print("\n[4/4] Testing module imports...")
    try:
        sys.path.insert(0, str(train_dir))
        from config import CVAEConfig
        from models import CVAE
        from dataset import fourier_encode_location
        
        config = CVAEConfig()
        print(f"✓ CVAEConfig loaded (device: {config.device})")
        print(f"✓ CVAE model architecture loaded")
        print(f"✓ Fourier encoding utility loaded")
        
        # Test Fourier encoding
        encoded = fourier_encode_location(40.7128, -74.0060)  # NYC
        print(f"✓ Fourier encoding test: {encoded.shape} features (NYC)")
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 60)
    print("\nYou're ready to train. Next steps:")
    print("  1. Ensure you've downloaded data:")
    print("     python scripts/download_metadata.py")
    print("     python scripts/download_filter_resumable.py")
    print()
    print("  2. Configure training (optional):")
    print("     edit train/config.py")
    print()
    print("  3. Start training:")
    print("     python train/train.py")
    print()
    print("  4. Monitor with TensorBoard (in separate terminal):")
    print("     tensorboard --logdir logs/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
