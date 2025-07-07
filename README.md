# Salps

Repository for scripts to run salp poop segmenting and analyze average color from masked regions.

## Overview

This project contains scripts for analyzing salp poop color characteristics from segmented images. The main script `mask_color.py` processes binary mask files and calculates color statistics for the masked regions.

## Prerequisites

- Python 3.6+
- Required packages (install via pip):
  ```bash
  pip install opencv-python numpy pandas tqdm
  ```

## Configuration

### Important: Update Base Directory

Before running the script, you **must** update the `base_dir` variable (I hard-coded it..) in `scripts/mask_color.py`:

```python
# Path to the project directory
base_dir = "/path/to/your/salp/project"  # CHANGE THIS TO YOUR PATH
```

The current path is hardcoded and will not work on other systems.

## Running the Script

### Input Requirements

The script expects the following directory structure:
```
your_project_directory/
├── session_*/                    # Session directories
│   └── filled_masks/            # Contains binary mask files
│       └── *_mask.png           # Mask files (white = poop, black = background)
├── OFP*/                        # Original image directories
│   └── *.JPG                    # Original images (must match mask names)
```

## Running the script
Just run the script with this command (make sure you're at the right level of the directory structure):
```
python3 mask_color.py
```

## Outputs

The script generates several types of output:

### 1. Output masks as a sanity check (On the first 3 processed images)
- **Location**: `debug_masked_areas/` directory
- **Files**: 
  - `debug_*_overlay.png` - Original image with red overlay on masked regions
  - `debug_*_diff.png` - Difference images (if comparing masks)

### 2. Color Analysis CSV Files
- **Location**: `session_*/salp_poop_color_summary.csv`
- **Contents**:
  - Session name
  - Mask and image filenames
  - Average RGB color values
  - Calculated brightness
  - Color category (Dark, Light, Reddish, Greenish, Bluish, Mixed)

### 3. Output on the command line
- Processing progress with tqdm
- Debug information for first 3 images
- Summary statistics including:
  - Brightness range and averages
  - Color category distribution
  - Most similar and most different color pairs
  - Session-wise analysis

## Color Analysis Features

### Brightness Calculation
Uses ITU-R Recommendation BT.601 standard:
```
Brightness = 0.299 * R + 0.587 * G + 0.114 * B
```

### Color Categorization
Simple RGB dominance-based categories:
- **Dark**: All RGB values < 50
- **Light**: All RGB values > 200
- **Reddish**: R > G and R > B
- **Greenish**: G > R and G > B
- **Bluish**: B > R and B > G
- **Mixed**: No clear dominance

### Color Similarity Analysis
- Calculates Euclidean distance between all color pairs
- Identifies most similar and most different colors
- Provides distance metrics for comparison

## Troubleshooting

### Common Issues

1. **"Missing image for [filename]"**
   - Ensure original images exist in OFP* directories
   - Check that image names match mask names (replace `_mask.png` with `.JPG`)

2. **"Error reading [file]"**
   - Verify file paths are correct
   - Check file permissions
   - Ensure images are in supported formats (JPG, PNG)

3. **"Mask is not binary"**
   - Masks should contain only 2 unique values (typically 0 and 255)
   - Check mask generation process

### Debug Mode
The script automatically generates debug information for the first 3 processed images, including:
- Mask analysis (shape, data type, value range)
- Pixel count statistics
- Overlay images for visual verification

## File Structure Example

```
your_project_directory/
├── session_001/
│   ├── filled_masks/
│   │   ├── image001_mask.png
│   │   └── image002_mask.png
│   └── salp_poop_color_summary.csv
├── session_002/
│   ├── filled_masks/
│   │   └── image003_mask.png
│   └── salp_poop_color_summary.csv
├── OFP_2023_01/
│   ├── image001.JPG
│   ├── image002.JPG
│   └── image003.JPG
└── debug_masked_areas/
    ├── debug_image001_overlay.png
    └── debug_image001_diff.png
```
