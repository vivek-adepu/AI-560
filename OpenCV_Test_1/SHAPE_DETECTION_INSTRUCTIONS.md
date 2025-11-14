# OpenCV Shape Detection - Instructions

## Overview

This project demonstrates how to detect and identify basic geometric shapes in images using OpenCV and Python. The system can recognize:
- Triangles
- Squares
- Rectangles
- Pentagons
- Circles

## How It Works

The shape detection algorithm uses **contour approximation** to identify shapes:

1. **Image Preprocessing**: Convert image to grayscale, apply Gaussian blur, and threshold
2. **Contour Detection**: Find contours (outlines) of shapes in the binary image
3. **Contour Approximation**: Reduce contour points using the Ramer-Douglas-Peucker algorithm
4. **Shape Classification**: Count vertices to determine shape type:
   - 3 vertices → Triangle
   - 4 vertices → Square or Rectangle (based on aspect ratio)
   - 5 vertices → Pentagon
   - More vertices → Circle

## Prerequisites

### Required Libraries

```bash
pip install opencv-python
pip install imutils
pip install numpy
```

### Optional (for Jupyter Notebook)

```bash
pip install jupyter
pip install matplotlib
```

## Project Structure

```
project/
│
├── pyimagesearch/
│   ├── __init__.py
│   └── shapedetector.py    # ShapeDetector class implementation
│
├── detect_shapes.py         # Main driver script
├── shapes_and_colors.png    # Sample image
└── OpenCV_shape_detect.ipynb # Jupyter notebook version
```

## Implementation Guide

### Step 1: Create the ShapeDetector Class

Create `pyimagesearch/shapedetector.py`:

```python
import cv2

class ShapeDetector:
    def __init__(self):
        pass
    
    def detect(self, c):
        # Initialize shape name
        shape = "unidentified"
        
        # Calculate perimeter
        peri = cv2.arcLength(c, True)
        
        # Approximate the contour
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Classify based on number of vertices
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            # Compute bounding box and aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # Square vs Rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        
        return shape
```

### Step 2: Create the Main Detection Script

Create `detect_shapes.py`:

```python
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Load and resize image
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# Preprocessing
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# Process each contour
for c in cnts:
    # Compute center
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    
    # Detect shape
    shape = sd.detect(c)
    
    # Scale contour back to original image size
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    
    # Draw contours and label
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

# Display results
cv2.imshow("Image", image)
cv2.waitKey(0)
```

## Usage

### Basic Usage

```bash
python detect_shapes.py --image shapes_and_colors.png
```

### Using Jupyter Notebook

1. Open the notebook:
   ```bash
   jupyter notebook OpenCV_shape_detect.ipynb
   ```

2. Run all cells sequentially

### Testing with Your Own Images

```bash
python detect_shapes.py --image /path/to/your/image.png
```

## Key Concepts Explained

### 1. Contour Approximation

The `cv2.approxPolyDP()` function uses the **Ramer-Douglas-Peucker algorithm** to reduce the number of points in a curve while maintaining its shape.

**Parameters:**
- `curve`: Input contour
- `epsilon`: Approximation accuracy (0.04 * perimeter works well)
- `closed`: Whether the curve is closed

### 2. Aspect Ratio Calculation

Used to distinguish squares from rectangles:

```python
aspect_ratio = width / height

# Square: AR ≈ 1.0 (0.95 - 1.05)
# Rectangle: AR significantly different from 1.0
```

### 3. Image Preprocessing Pipeline

```
Original Image → Grayscale → Gaussian Blur → Threshold → Binary Image
```

This pipeline:
- Reduces noise (blur)
- Simplifies the image (grayscale)
- Separates shapes from background (threshold)

## Advanced Techniques

### Using Hu Moments

For more robust shape recognition:

```python
import cv2

# Calculate Hu moments
moments = cv2.moments(contour)
hu_moments = cv2.HuMoments(moments)
```

### Using Zernike Moments

Provides 25 features for shape description (requires `mahotas` library):

```python
import mahotas

# Calculate Zernike moments
zernike = mahotas.features.zernike_moments(image, radius=21)
```

### Deep Learning Approach

For complex scenarios, consider using CNNs:
- Better accuracy
- Handles rotations and deformations
- Requires training data
- Use frameworks like TensorFlow or PyTorch

## Troubleshooting

### Shapes Not Detected

**Issue**: No shapes found
**Solutions:**
- Adjust threshold value (try 50, 60, 70, 100)
- Check if image has sufficient contrast
- Ensure shapes are distinct from background

### Incorrect Shape Classification

**Issue**: Shapes labeled incorrectly
**Solutions:**
- Adjust epsilon value in `approxPolyDP()` (try 0.02 to 0.06)
- Modify aspect ratio range for square detection
- Ensure image preprocessing is adequate

### Multiple Contours per Shape

**Issue**: One shape has multiple detections
**Solutions:**
- Use `cv2.RETR_EXTERNAL` to get only outer contours
- Apply morphological operations (erosion/dilation)
- Filter contours by area/perimeter

## Performance Optimization

### For Faster Processing

```python
# Resize image to smaller dimensions
resized = imutils.resize(image, width=300)

# Use simpler contour approximation
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### For Better Accuracy

```python
# Use bilateral filter instead of Gaussian blur
blurred = cv2.bilateralFilter(gray, 11, 17, 17)

# Adaptive thresholding for varying lighting
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
```

## Example Results

When run successfully, you should see:
- Green contours around detected shapes
- White text labels indicating shape type (triangle, square, rectangle, pentagon, circle)
- Labels positioned at the center of each shape

## Further Learning

### Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyImageSearch Blog](https://pyimagesearch.com)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### Next Steps
1. Add color detection to identify shape colors
2. Implement rotation-invariant detection
3. Handle overlapping shapes
4. Add shape tracking in video streams
5. Build a shape classification dataset for ML

## Credits

Based on the tutorial by Adrian Rosebrock at PyImageSearch:
- Original Article: [OpenCV Shape Detection](https://pyimagesearch.com/2016/02/08/opencv-shape-detection/)
- Published: February 8, 2016
- Updated: July 7, 2021

## License

Refer to the original PyImageSearch tutorial for licensing information.

---

**Happy Shape Detecting! 🔺🔵🔷**
