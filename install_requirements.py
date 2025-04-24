import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Installing dependencies for neural style transfer project...\n")

    packages = [
        "torch",             # PyTorch core
        "torchvision",       # Pretrained models (VGG, ResNet, AlexNet)
        "torchaudio",        # Optional: Safe to install with torch
        "opencv-python",     # Image loading and processing (cv2)
        "Pillow",            # Image saving
        "matplotlib",        # Optional: visualization
        "numpy",             # Core scientific computing
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        install(pkg)

    print("\n All dependencies installed successfully.")

if __name__ == "__main__":
    main()

# python install_requirements.py
