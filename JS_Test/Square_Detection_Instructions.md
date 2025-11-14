# Square Detection System - Instructions

## Overview

The Square Detection System is a browser-based computer vision tool that automatically identifies square-shaped objects in images and extracts their center coordinates. It uses connected component labeling algorithms to detect individual shapes and filters them based on size and aspect ratio.

## Features

- **Automatic Square Detection**: Identifies square-like shapes in uploaded images
- **Visual Feedback**: Displays detected squares with green bounding boxes and red center points
- **Adjustable Parameters**: Customizable minimum and maximum size thresholds
- **JSON Export**: Download detection results as structured JSON data
- **Real-time Processing**: Browser-based processing with immediate visual feedback

## How to Use

### Getting Started

1. **Open the Application**
   - Double-click `Shape_square_detection.html` to open in your web browser
   - No installation or setup required - runs entirely in the browser

2. **Upload an Image**
   - Click the "Upload Image" button
   - Select an image file (PNG, JPG, or other common formats)
   - The image will display on the canvas

3. **Adjust Detection Parameters**
   - **Min Size**: Minimum size in pixels for detected squares (default: 3px)
   - **Max Size**: Maximum size in pixels for detected squares (default: 50px)
   - Adjust these sliders before detection to filter results

4. **Run Detection**
   - Click the "Detect Squares" button
   - Processing typically takes less than a second
   - Results will overlay on the image

5. **Review Results**
   - Green boxes show detected square boundaries
   - Red dots and crosshairs mark center points
   - Scroll through the list below to see coordinates for each square
   - Total count displayed in the "Detection Results" section

6. **Export Data**
   - Click "Download JSON" to save detection results
   - JSON file includes image dimensions, square count, and detailed data for each square

## Understanding the Detection Algorithm

### How It Works

1. **Binary Conversion**: Converts the image to black and white using a brightness threshold (128)
2. **Connected Component Labeling**: Groups connected white pixels into distinct regions
3. **Bounding Box Calculation**: Determines the rectangular boundary for each region
4. **Shape Filtering**: Keeps only regions that meet the criteria:
   - Size between min and max parameters
   - Aspect ratio between 0.7 and 1.4 (nearly square)
5. **Center Calculation**: Computes the geometric center of each validated square

### What Gets Detected

The system identifies shapes that are:
- Composed of connected white pixels on a black background
- Roughly square-shaped (aspect ratio near 1:1)
- Within the specified size range
- Distinct from other shapes (not overlapping)

## JSON Output Format

```json
{
  "imageSize": {
    "width": 280,
    "height": 440
  },
  "squaresDetected": 89,
  "squares": [
    {
      "center": {
        "x": 13.5,
        "y": 13.5
      },
      "bounds": {
        "x": 9,
        "y": 9,
        "width": 10,
        "height": 10
      },
      "size": 10,
      "aspectRatio": "1.00",
      "pixelCount": 100
    }
    // ... more squares
  ]
}
```

### Field Descriptions

- **imageSize**: Original image dimensions in pixels
- **squaresDetected**: Total number of squares found
- **squares**: Array of detected squares, each containing:
  - **center**: X,Y coordinates of the square's center point
  - **bounds**: Bounding box (top-left corner x,y and width/height)
  - **size**: Maximum dimension (width or height)
  - **aspectRatio**: Width divided by height
  - **pixelCount**: Total number of white pixels in the shape

## Tips for Best Results

### Image Preparation

- **Use high contrast images**: Black background with white squares works best
- **Clear separation**: Ensure squares don't touch or overlap
- **Appropriate resolution**: Match image size to your square sizes
- **Clean edges**: Crisp, well-defined edges improve detection accuracy

### Parameter Tuning

- **Small squares (1-5px)**: Set Min=1, Max=10
- **Medium squares (5-20px)**: Set Min=3, Max=30 (default range)
- **Large squares (20-100px)**: Set Min=10, Max=150
- **Mixed sizes**: Use wider range but expect more false positives

### Common Issues

**Too many false detections?**
- Narrow the size range
- Use cleaner source images
- Adjust aspect ratio tolerance in code (currently 0.7-1.4)

**Missing valid squares?**
- Expand the size range
- Check if squares are connected (touching squares merge)
- Verify image contrast is sufficient

**Irregular shapes detected?**
- Current aspect ratio filter (0.7-1.4) allows slightly rectangular shapes
- For stricter square detection, modify the aspectRatio threshold in code

## Test Images Included

The project includes several test images demonstrating different use cases:

1. **test_7_14.png** (280×440): Grid pattern with 89 squares, mixed sizes
2. **test_12_12.png** (450×450): Regular grid of 12×12 uniform squares
3. **test_26_30.png** (480×480): Dense pattern with 440+ small squares
4. **test_21_4.png** (480×96): Horizontal arrangement of 21 squares
5. **test_38_38.png** (480×480): Large grid of 1444 tiny squares
6. **test_30_30.png** (480×480): Dense 900-square grid

Try these to understand how parameters affect detection!

## Technical Details

### Technologies Used

- **React 18**: UI framework with hooks for state management
- **HTML5 Canvas**: Image rendering and overlay graphics
- **Babel Standalone**: In-browser JSX compilation
- **Tailwind CSS**: Responsive styling
- **Vanilla JavaScript**: Core detection algorithms

### Algorithm Complexity

- **Time Complexity**: O(W × H) where W and H are image dimensions
- **Space Complexity**: O(W × H) for pixel data storage
- **Typical Performance**: <1 second for images under 1000×1000 pixels

### Browser Compatibility

Works in all modern browsers:
- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

Requires JavaScript enabled and canvas support.

## Customization Options

### Code Modifications

To adjust detection sensitivity, edit these values in the code:

```javascript
// Brightness threshold for binary conversion (line ~104)
binary[i / 4] = brightness > 128 ? 1 : 0;  // Change 128

// Aspect ratio tolerance (line ~152)
aspectRatio > 0.7 && aspectRatio < 1.4  // Adjust 0.7 and 1.4

// Default size parameters (line ~22-23)
const [minSize, setMinSize] = useState(3);  // Default minimum
const [maxSize, setMaxSize] = useState(50); // Default maximum
```

### Visual Customization

Modify overlay colors:

```javascript
// Bounding box color (line ~176)
ctx.strokeStyle = '#00ff00';  // Green boxes

// Center point color (line ~185)
ctx.fillStyle = '#ff0000';    // Red centers
```

## Use Cases

- **Quality Control**: Automated inspection of manufactured patterns
- **Design Analysis**: Quantifying square elements in textile or graphic designs
- **Data Extraction**: Converting visual patterns to coordinate data
- **Education**: Teaching computer vision concepts
- **Research**: Pattern recognition studies

## Support & Troubleshooting

### File Won't Open
- Ensure you're opening the HTML file in a web browser
- Try a different browser if issues persist
- Check that JavaScript is enabled

### No Squares Detected
- Verify image has white squares on black background
- Adjust min/max size parameters
- Check that squares are separated (not touching)

### Performance Issues
- Use smaller images (under 2000×2000 pixels recommended)
- Close other browser tabs
- Try a different browser

## Credits

Built using connected component labeling algorithms for robust shape detection. Designed for ease of use with no dependencies beyond standard web technologies.

---

**Version**: 1.0  
**Last Updated**: November 2025  
**License**: Free to use and modify
