import os
import numpy as np
from PIL import Image

def run_dataset_processing(root_path="./hiragana", threshold=128):
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
                    img = img.resize((128, 128))
                    img = img.convert("L").point(lambda p: 255 if p > threshold else 0).convert("1")
                    
                    # 2. Get bits as a list of integers (0 or 1)
                    bits = [1 if p > 0 else 0 for p in img.getdata()]
                    
                    # 3. Append to our master lists
                    all_pixels.append(bits)
                    all_labels.append(label)

    # Convert the list of lists into a 2D NumPy array
    x_raw = np.array(all_pixels, dtype=np.uint8)
    y_raw = all_labels

    return x_raw, y_raw