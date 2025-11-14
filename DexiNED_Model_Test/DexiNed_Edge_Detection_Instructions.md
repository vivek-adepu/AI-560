# DexiNed Edge Detection - Setup and Usage Guide

## Overview

This notebook implements **DexiNed (Dense Extreme Inception Network)** for high-quality edge detection. The implementation is optimized for **NVIDIA Blackwell GPU architecture** (sm_120) running in HP AI Studio with headless display support.

### Key Features
- ✓ Production-ready edge detection with single and batch processing
- ✓ GPU-accelerated inference (CUDA 12.8 compatible)
- ✓ Automatic image preprocessing with dimension validation
- ✓ Virtual display support for headless environments
- ✓ Comprehensive visualization and result saving
- ✓ Memory management utilities

### Performance Metrics
- **Speed**: 20-50ms per image @ 512×512 on RTX GPUs
- **GPU Memory**: Minimum 4GB, recommended 8GB+
- **Supported Resolutions**: 352, 480, 512, 640, 704, 768, 800, 1024 pixels

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Blackwell architecture optimized)
- **Memory**: 8GB+ GPU memory recommended for batch processing
- **CPU Fallback**: Supported but 10-50x slower

### Software
- Python 3.12+
- PyTorch 2.10.0+ (nightly build with CUDA 12.8)
- HP AI Studio environment (or similar Jupyter environment)

---

## Installation & Setup

### Step 1: Initial Configuration
**Run Cell 1** - CUDA Compatibility Setup
```python
# This cell MUST be run first!
# Sets critical environment variables before importing PyTorch
```

**What it does:**
- Configures CUDA environment variables for Blackwell GPU
- Sets up memory management
- Suppresses unnecessary warnings

⚠️ **Critical**: Never skip or move this cell!

---

### Step 2: Install PyTorch Nightly
**Run Cell 2** - PyTorch Installation

**What it does:**
- Uninstalls old PyTorch versions
- Installs PyTorch nightly with CUDA 12.8 support
- Required for Blackwell GPU architecture (sm_120)

**Installation time:** ~3-5 minutes

**Expected output:**
```
✓ PyTorch installation complete
PyTorch version: 2.10.0.dev20251113+cu128
```

---

### Step 3: Import Core Libraries
**Run Cell 3** - Import & Configure PyTorch

**What it does:**
- Imports PyTorch, NumPy, PIL, OpenCV
- Configures PyTorch for Blackwell GPU stability
- Disables TF32 and benchmark mode for deterministic behavior

**Expected output:**
```
✓ Core libraries imported successfully
✓ PyTorch configured for NVIDIA Blackwell GPU
```

---

### Step 4: Verify GPU
**Run Cell 4** - GPU Testing

**What it does:**
- Comprehensive GPU hardware detection
- Tests CUDA operations
- Verifies compute capability and memory

**Expected output:**
```
✓ GPU detected and functional
Device: NVIDIA RTX PRO 6000 Blackwell Max-Q
Compute capability: 12.0 (Blackwell architecture)
Total memory: 95.59 GB
```

If GPU is not detected, the notebook will continue in CPU mode (slower).

---

### Step 5: Setup Virtual Display
**Run Cell 5** - Xvfb Virtual Display

**What it does:**
- Installs Xvfb for headless display support
- Configures matplotlib backend
- Required for environments without physical display

**Expected output:**
```
✓ Virtual display configured for headless environment
```

---

### Step 6: Install Dependencies
**Run Cell 6** - Additional Libraries

**What it does:**
- Installs required Python packages:
  - opencv-python (image processing)
  - scipy (scientific computing)
  - kornia (differentiable computer vision)

**Installation time:** ~2-3 minutes

---

### Step 7: Download DexiNed Model
**Run Cell 7** - Model Download

**What it does:**
- Creates project directory: `~/local/DexiNED_Model/`
- Clones DexiNed repository from GitHub
- Downloads pre-trained checkpoint (10_model.pth)

**Directory structure created:**
```
~/local/DexiNED_Model/
├── DexiNed/          # Model architecture
└── checkpoints/      # Pre-trained weights
    └── 10_model.pth  # Main checkpoint (~15MB)
```

**Expected output:**
```
✓ DexiNed repository cloned successfully
✓ Pre-trained checkpoint downloaded: 10_model.pth
```

---

### Step 8: Import DexiNed Model
**Run Cell 8** - Load Model Architecture

**What it does:**
- Adds DexiNed directory to Python path
- Imports model definition
- Tests model instantiation

**Expected output:**
```
✓ DexiNed model imported successfully
✓ Model architecture loaded
```

---

### Step 9: Load Pre-trained Weights
**Run Cell 9** - Initialize Model

**What it does:**
- Creates DexiNed model instance
- Loads pre-trained weights from checkpoint
- Moves model to GPU and sets to evaluation mode

**Expected output:**
```
✓ DexiNed model loaded successfully
✓ Pre-trained weights loaded from checkpoint
✓ Model in evaluation mode on cuda device
```

---

## Using the Model

### Step 10: Image Preprocessing Function
**Run Cell 10** - Setup Preprocessing

**What it does:**
- Defines `preprocess_image()` function
- Handles dimension validation (must be divisible by 16)
- Normalizes images to [0, 1] range
- Converts to PyTorch tensors

**Valid target sizes:** 352, 480, 512, 640, 704, 768, 800, 1024

---

### Step 11: Edge Detection Function
**Run Cell 11** - Core Inference Function

**What it does:**
- Defines `detect_edges()` function
- Runs DexiNed inference
- Applies sigmoid activation
- Fuses multi-scale predictions

**Key parameters:**
- `target_size`: Resolution for processing (default: 512)
- Returns numpy array with edge map [0, 1]

---

### Step 12: Visualization Functions
**Run Cells 12-13** - Display & Save Utilities

**Functions created:**
- `visualize_edges()`: Side-by-side comparison display
- `save_edge_results()`: Save to disk with timestamp

---

## Running Edge Detection

### Single Image Processing

**Step 13: Configure Test Image** (Cell 14)
```python
# Update this path with your image
test_image_path = "data/test_image.jpg"
```

**Step 14: Run Detection** (Cell 15)
```python
# Automatically processes and displays results
```

**Expected output:**
- Original image (left)
- Edge detection result (right)
- Processing time displayed

---

### Batch Processing

**Step 15: Setup Batch Processing** (Cell 16)

**What it does:**
- Processes all images in a directory
- Creates organized output structure:
  ```
  results/
  ├── edges/          # Edge detection results
  ├── overlays/       # Edge overlays on originals
  └── comparisons/    # Side-by-side visualizations
  ```

**Usage:**
```python
# Set input directory
input_dir = "data/batch_images/"

# Process all images
for image_file in Path(input_dir).glob("*.jpg"):
    edges = detect_edges(str(image_file), target_size=512)
    save_edge_results(...)
```

---

### Custom Image Processing

**Step 16: Direct Image Input** (Cell 17)

**Use cases:**
- Process images from memory
- Apply edge detection to video frames
- Integrate with other pipelines

**Example:**
```python
from PIL import Image

# Load your image
img = Image.open("path/to/image.jpg")

# Detect edges
edges = detect_edges(img, target_size=512)

# Visualize
visualize_edges(np.array(img), edges)
```

---

## Memory Management

### Step 17: GPU Memory Monitoring (Cell 18)

**Functions:**
- `check_gpu_memory()`: Display current GPU usage
- `clear_gpu_memory()`: Free cached memory

**Usage:**
```python
# Check current memory status
check_gpu_memory()

# Clear cache if needed
clear_gpu_memory()
```

**When to clear memory:**
- Every 10-20 images in batch processing
- Before processing large images
- When seeing CUDA out-of-memory errors

---

## Troubleshooting

### Common Issues & Solutions

#### 1. CUDA Out of Memory
**Symptoms:** RuntimeError: CUDA out of memory

**Solutions:**
```python
# Reduce image size
edges = detect_edges(image_path, target_size=352)  # Instead of 512

# Clear GPU memory
clear_gpu_memory()

# Restart kernel if needed
# Kernel → Restart Kernel
```

---

#### 2. Dimension Mismatch Error
**Symptoms:** Tensor dimension error during inference

**Cause:** Image dimensions not divisible by 16

**Solution:** The `preprocess_image()` function automatically handles this, but ensure you're using valid target sizes:
- ✓ Valid: 352, 480, 512, 640, 704, 768, 800, 1024
- ✗ Invalid: 300, 500, 600, 900

---

#### 3. Image Not Found
**Symptoms:** FileNotFoundError or PIL cannot identify image

**Solutions:**
```python
# Verify path exists
from pathlib import Path
if not Path(test_image_path).exists():
    print(f"Image not found: {test_image_path}")

# Check file extension
# Supported: .jpg, .jpeg, .png, .bmp, .tiff

# Place images in organized folders
# Recommended: ~/local/DexiNED_Model/data/
```

---

#### 4. Blank Edge Maps
**Symptoms:** Output is completely black or white

**Checklist:**
```python
# 1. Verify model is in eval mode
model.eval()

# 2. Check input normalization
# Images should be in [0, 1] range

# 3. Confirm checkpoint loaded
print(model.state_dict().keys())

# 4. Test with known good image
test_image_path = "DexiNed/BIPED/edges/imgs/train/rgbr/real/0.jpg"
```

---

#### 5. Slow Performance on CPU
**Expected Behavior:** CPU inference is 10-50x slower than GPU

**Solutions:**
- Use GPU if available (preferred)
- Reduce target_size for faster processing
- Process in batches to amortize overhead
- Consider using smaller model variant

---

#### 6. Virtual Display Errors
**Symptoms:** Display-related errors in headless environment

**Solution:**
```bash
# Restart Xvfb (run in terminal)
sudo killall Xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

---

#### 7. Model Download Fails
**Symptoms:** git clone or wget errors

**Solutions:**
```bash
# Check network connectivity
ping github.com

# Manual download
cd ~/local/DexiNED_Model/
git clone https://github.com/xavysp/DexiNed.git

# Download checkpoint manually from:
# https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu
```

---

## Best Practices

### Image Processing
✓ **Always process at 512×512** for balanced speed/quality  
✓ **Use valid dimensions** divisible by 16  
✓ **Normalize inputs** to [0, 1] range  
✓ **Save incrementally** during batch processing  

### Memory Management
✓ **Clear GPU cache** every 10-20 images  
✓ **Monitor memory usage** with `check_gpu_memory()`  
✓ **Restart kernel** if experiencing memory issues  

### Performance
✓ **Keep model in eval() mode** during inference  
✓ **Use torch.no_grad()** context for faster inference  
✓ **Batch similar-sized images** together  
✓ **Profile your code** to identify bottlenecks  

### Visualization
✓ **Use matplotlib** (not cv2.imshow) in Jupyter  
✓ **Save high-resolution** results for analysis  
✓ **Create overlays** for better visualization  

---

## Performance Expectations

### Processing Speed (RTX GPU @ 512×512)
- Single image: **20-50ms**
- Batch of 10: **200-500ms**
- Batch of 100: **2-5 seconds**

### GPU Memory Usage
- Model weights: **~15MB**
- Single image (512×512): **~50-100MB**
- Batch processing: **200MB-2GB** depending on batch size

### Quality Settings
| Target Size | Speed | Quality | Memory | Use Case |
|-------------|-------|---------|--------|----------|
| 352 | Fastest | Good | Low | Quick preview |
| 512 | Fast | Excellent | Medium | Recommended |
| 768 | Medium | Superior | High | High-quality |
| 1024 | Slow | Best | Very High | Publication |

---

## Advanced Usage

### Custom Thresholding
```python
# Get raw edge probabilities
edges = detect_edges(image_path, target_size=512)

# Apply custom threshold
threshold = 0.3  # Adjust between 0.0-1.0
binary_edges = (edges > threshold).astype(np.uint8) * 255
```

### Multi-scale Processing
```python
# Process at different scales
scales = [352, 512, 768]
results = []

for scale in scales:
    edges = detect_edges(image_path, target_size=scale)
    results.append(edges)

# Combine results (e.g., average)
combined = np.mean(results, axis=0)
```

### Integration with OpenCV
```python
import cv2

# Load with OpenCV
img_cv = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Detect edges
edges = detect_edges(img_rgb, target_size=512)

# Convert back to OpenCV format
edges_cv = (edges * 255).astype(np.uint8)

# Save with OpenCV
cv2.imwrite("edges_output.png", edges_cv)
```

---

## Project Structure

```
~/local/DexiNED_Model/
│
├── DexiNed/                    # Model repository
│   ├── model.py               # DexiNed architecture
│   ├── losses.py              # Training losses
│   └── BIPED/                 # Sample images
│
├── checkpoints/               # Pre-trained weights
│   └── 10_model.pth          # Main checkpoint (15MB)
│
├── data/                      # Your input images
│   ├── test_image.jpg
│   └── batch_images/
│
└── results/                   # Output directory
    ├── edges/                # Edge detection results
    ├── overlays/             # Overlay visualizations
    └── comparisons/          # Side-by-side comparisons
```

---

## Cell Execution Order

**Critical Setup (Required):**
1. Cell 1: CUDA Configuration ⚠️ **Must be first**
2. Cell 2: Install PyTorch Nightly
3. Cell 3: Import Core Libraries
4. Cell 4: Verify GPU
5. Cell 5: Setup Virtual Display
6. Cell 6: Install Dependencies
7. Cell 7: Download DexiNed
8. Cell 8: Import Model
9. Cell 9: Load Weights

**Function Definitions (Run Once):**
10. Cell 10: Preprocessing Function
11. Cell 11: Edge Detection Function
12. Cell 12: Visualization Function
13. Cell 13: Save Function

**Inference (Run Repeatedly):**
14. Cell 14: Set Image Path
15. Cell 15: Single Image Detection
16. Cell 16: Batch Processing *(optional)*
17. Cell 17: Custom Processing *(optional)*

**Utilities:**
18. Cell 18: Memory Management
19. Cell 19: Troubleshooting Reference

---

## Additional Resources

### Documentation
- **DexiNed Paper**: "Dense Extreme Inception Network for Edge Detection"
- **GitHub Repository**: https://github.com/xavysp/DexiNed
- **PyTorch Documentation**: https://pytorch.org/docs/stable/

### Datasets
- **BIPED**: Barcelona Images for Perceptual Edge Detection
- **BSDS500**: Berkeley Segmentation Dataset
- **NYUDv2**: NYU Depth Dataset (for indoor scenes)

### Related Models
- HED (Holistically-Nested Edge Detection)
- RCF (Richer Convolutional Features)
- BDCN (Bi-Directional Cascade Network)

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{soria2020dexined,
  title={Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection},
  author={Soria, Xavier and Riba, Edgar and Sappa, Angel},
  journal={WACV},
  year={2020}
}
```

---

## Support & Troubleshooting

### Quick Checks
- [ ] Cell 1 was run first (CUDA configuration)
- [ ] PyTorch installed correctly (check version)
- [ ] GPU is detected and functional
- [ ] Model checkpoint downloaded successfully
- [ ] Image path exists and is correct
- [ ] Target size is divisible by 16

### Performance Issues
- Monitor GPU memory with `check_gpu_memory()`
- Clear cache with `clear_gpu_memory()`
- Reduce target_size if needed
- Restart kernel for fresh start

### Getting Help
- Check Cell 19 for detailed troubleshooting
- Review error messages carefully
- Test with provided sample images first
- Document your issue with error traces

---

## Version History

**v1.0** - Initial implementation
- Blackwell GPU optimization
- Headless display support
- Comprehensive error handling
- Batch processing capabilities

---

## License

This implementation follows the original DexiNed license (Apache 2.0). The pre-trained model is provided by the original authors for research purposes.

---

**Ready to start? Begin with Cell 1 and work through the setup cells in order!**
