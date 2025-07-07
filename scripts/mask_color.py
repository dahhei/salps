import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd

# Path to the project directory
base_dir = "/Users/choij/Desktop/git_repositories/Salp_Project"

# Create output directory for debug images
debug_dir = os.path.join(base_dir, "debug_masked_areas")
os.makedirs(debug_dir, exist_ok=True)

# Find all mask files in any `filled_masks/` subfolder
mask_files = glob(os.path.join(base_dir, "session_*/filled_masks/*_mask.png"))

# finding the original images
image_dirs = [
    os.path.join(base_dir, d) for d in os.listdir(base_dir)
    if d.startswith("OFP") and os.path.isdir(os.path.join(base_dir, d))
]

def calculate_brightness(r, g, b):
    """Calculate perceived brightness using luminance formula
    This comes from the ITU-R Recommendation BT.601 standard (found online for brightness)
    """
    return 0.299 * r + 0.587 * g + 0.114 * b

def get_simple_color_category(r, g, b):
    """Simple color categorization based on RGB dominance to say what color the poop is closest to"""
    if max(r, g, b) < 50:
        return "Dark"
    elif min(r, g, b) > 200:
        return "Light"
    elif r > g and r > b:
        return "Reddish"
    elif g > r and g > b:
        return "Greenish"
    elif b > r and b > g:
        return "Bluish"
    else:
        return "Mixed"

def calculate_color_distance(color1, color2):
    """Calculate Euclidean distance between two RGB colors to find the most similar and most different colors"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

# Function to find original image for a given mask name
def find_image(mask_filename):
    img_name = mask_filename.replace("_mask.png", ".JPG")
    for folder in image_dirs:
        path = os.path.join(folder, img_name)
        if os.path.exists(path):
            return path
    return None


results = []

for mask_path in tqdm(mask_files):
    mask_filename = os.path.basename(mask_path)
    session_name = mask_path.split("/")[-3]
    image_path = find_image(mask_filename)
    if not image_path:
        print(f"Missing image for {mask_filename}")
        continue
    
    # Load files
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"Error reading {mask_path} or {image_path}")
        continue
    
    # Debug mask analysis for first few images
    if len(results) < 3:
        print(f"\n=== MASK ANALYSIS: {mask_filename} ===")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask data type: {mask.dtype}")
        print(f"Mask value range: {mask.min()} to {mask.max()}")
        
        # Count different pixel values in mask
        unique_values, counts = np.unique(mask, return_counts=True)
        print(f"Unique values in mask: {unique_values}")
        print(f"Counts: {counts}")
        
        # Verify it's binary
        if len(unique_values) != 2:
            print(f"WARNING: Mask is not binary! Has {len(unique_values)} values instead of 2")
        else:
            print(f"✓ Mask is binary with values: {unique_values}")
    
    # Apply mask - use exactly what's in the binary mask
    binary_mask = mask > 0
    poop_pixels = img[binary_mask]
    
    # Debug info and save example images for first few images
    if len(results) < 3:  # Only show for first 3 images to avoid spam
        total_pixels = img.shape[0] * img.shape[1]
        masked_pixels = len(poop_pixels)
        print(f"Debug: {mask_filename}")
        print(f"  Total image pixels: {total_pixels}")
        print(f"  Masked poop pixels: {masked_pixels} ({masked_pixels/total_pixels*100:.1f}% of image)")
        
        # Reload the image and mask to ensure they are unmodified
        img_overlay = cv2.imread(image_path)
        mask_overlay = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        overlay = img_overlay.copy()
        overlay[mask_overlay > 0] = [0, 0, 255]  # Red overlay
        overlay_path = os.path.join(debug_dir, f"debug_{mask_filename.replace('_mask.png', '_overlay.png')}")
        cv2.imwrite(overlay_path, overlay)
        print(f"Overlay saved to: {overlay_path}")

    # Print file paths for debugging
    print(f"\n--- Debug Info ---")
    print(f"Image file: {image_path}")
    print(f"Mask file: {mask_path}")

    # Print unique values and shapes
    print("Mask unique values:", np.unique(mask))
    print("Mask shape:", mask.shape)
    print("Image shape:", img.shape[:2])

    # If you want to compare two mask files pixel-by-pixel, specify the second mask path here
    # For demonstration, let's assume you want to compare the current mask to a reference mask in filled_masks
    reference_mask_path = os.path.join(os.path.dirname(mask_path), mask_filename)
    if os.path.exists(reference_mask_path):
        reference_mask = cv2.imread(reference_mask_path, cv2.IMREAD_GRAYSCALE)
        if reference_mask is not None:
            print("Reference mask unique values:", np.unique(reference_mask))
            print("Reference mask shape:", reference_mask.shape)
            identical = np.array_equal(mask, reference_mask)
            print(f"Mask identical to reference: {identical}")
            if not identical:
                diff = np.abs(mask.astype(np.int16) - reference_mask.astype(np.int16))
                diff_pixels = np.count_nonzero(diff)
                print(f"Number of differing pixels: {diff_pixels}")
                # Save a diff image for visual inspection
                diff_img = np.zeros_like(mask)
                diff_img[diff != 0] = 255
                diff_path = os.path.join(debug_dir, f"debug_{mask_filename.replace('_mask.png', '_diff.png')}")
                cv2.imwrite(diff_path, diff_img)
                print(f"Diff image saved to: {diff_path}")
        else:
            print(f"Reference mask could not be loaded: {reference_mask_path}")
    else:
        print(f"Reference mask not found: {reference_mask_path}")

    if poop_pixels.size == 0:
        avg_color = [0, 0, 0]
        brightness = 0
        color_category = "Dark"
    else:
        avg_color = np.mean(poop_pixels, axis=0).tolist()  # BGR
        # Convert BGR to RGB for analysis
        r, g, b = avg_color[2], avg_color[1], avg_color[0]
        brightness = calculate_brightness(r, g, b)
        color_category = get_simple_color_category(r, g, b)
    
    results.append({
        "session": session_name,
        "mask_file": mask_filename,
        "image_file": os.path.basename(image_path),
        "avg_color_b": round(avg_color[0], 2),
        "avg_color_g": round(avg_color[1], 2),
        "avg_color_r": round(avg_color[2], 2),
        "brightness": round(brightness, 2),
        "color_category": color_category
    })

# Save a separate CSV for each session
df = pd.DataFrame(results)

for session_name, group in df.groupby('session'):
    session_dir = os.path.join(base_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    out_csv = os.path.join(session_dir, "salp_poop_color_summary.csv")
    group.to_csv(out_csv, index=False)
    print(f"Saved results for session {session_name} to {out_csv}")

# Print summary statistics
print(f"\nProcessed {len(results)} mask files")
print(f"\nColor Analysis Summary:")
print(f"Brightness range: {df['brightness'].min():.1f} - {df['brightness'].max():.1f}")
print(f"Average brightness: {df['brightness'].mean():.1f}")

print(f"\nColor Categories:")
category_counts = df['color_category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} ({count/len(results)*100:.1f}%)")

# Find most similar and most different colors
print(f"\nColor Similarity Analysis:")
if len(results) > 1:
    # Calculate all pairwise distances
    colors = df[['avg_color_r', 'avg_color_g', 'avg_color_b']].values
    min_distance = float('inf')
    max_distance = 0
    most_similar_pair = None
    most_different_pair = None
    
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            distance = calculate_color_distance(colors[i], colors[j])
            if distance < min_distance:
                min_distance = distance
                most_similar_pair = (i, j)
            if distance > max_distance:
                max_distance = distance
                most_different_pair = (i, j)
    
    if most_similar_pair:
        idx1, idx2 = most_similar_pair
        print(f"Most similar colors:")
        print(f"  {df.iloc[idx1]['mask_file']} (RGB: {colors[idx1][0]:.0f},{colors[idx1][1]:.0f},{colors[idx1][2]:.0f})")
        print(f"  {df.iloc[idx2]['mask_file']} (RGB: {colors[idx2][0]:.0f},{colors[idx2][1]:.0f},{colors[idx2][2]:.0f})")
        print(f"  Distance: {min_distance:.1f}")
    
    if most_different_pair:
        idx1, idx2 = most_different_pair
        print(f"\nMost different colors:")
        print(f"  {df.iloc[idx1]['mask_file']} (RGB: {colors[idx1][0]:.0f},{colors[idx1][1]:.0f},{colors[idx1][2]:.0f})")
        print(f"  {df.iloc[idx2]['mask_file']} (RGB: {colors[idx2][0]:.0f},{colors[idx2][1]:.0f},{colors[idx2][2]:.0f})")
        print(f"  Distance: {max_distance:.1f}")

# Session-wise analysis
print(f"\nSession-wise Analysis:")
session_stats = df.groupby('session').agg({
    'brightness': ['mean', 'std', 'min', 'max'],
    'color_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
}).round(2)
print(session_stats)
