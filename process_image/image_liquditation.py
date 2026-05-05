import os
from PIL import Image

def run_dataset_processing(root_path="./hiragana", threshold=128):
    """
    Crawls all folders, processes every image, 
    and returns a dictionary of {character: [list_of_bit_strings]}
    """
    all_data = {}

    for subdir, dirs, files in os.walk(root_path):
        # The folder name (e.g., 'aa', 'ki') is our label
        label = os.path.basename(subdir)
        
        # Skip the root folder itself
        if subdir == root_path:
            continue

        all_data[label] = []

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, filename)
                
                # Process the pixels
                with Image.open(file_path) as img:
                    img = img.resize((128, 128))
                    img = img.convert("L").point(lambda p: 255 if p > threshold else 0).convert("1")
                    
                    # Convert to string of 0s and 1s
                    bits = "".join(['1' if p > 0 else '0' for p in img.getdata()])
                    all_data[label].append(bits)
    
    return all_data