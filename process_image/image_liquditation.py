import os
import numpy as np
from PIL import Image
from settings import DATA_DIR, IMAGE_SIZE


def run_dataset_processing(root_path=DATA_DIR):
    """
    Crawls folders and returns x_raw (NumPy array) and y_raw (list of labels).
    """
    all_pixels = []
    all_labels = []

    # walk() through the directory
    for subdir, dirs, files in os.walk(root_path):
        label = os.path.basename(subdir)
        
        # Skip the root folder itself
        if subdir == root_path:
            continue

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, filename)
                
                with Image.open(file_path) as img:
                    # 1. Resize and convert to B&W
                    img = img.resize(IMAGE_SIZE)
                    img = img.convert("L")

                    # 3. Append flattened, normalized pixels to all_pixels
                    all_pixels.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
                    
                    # 3. Append the folder name as the label
                    all_labels.append(label)

    # Store flattened, normalized image pixels as a 2D array:
    # one row per image, one column per pixel
    # values are normalized grayscale intensities in the range [0.0, 1.0]
    x_raw = np.array(all_pixels, dtype=np.float32)
    y_raw = all_labels

    return x_raw, y_raw