from process_image.image_liquditation import run_dataset_processing

if __name__ == "__main__":
    # The function handles the loop, the folders, and the image processing
    hiragana_dataset = run_dataset_processing()

    # Just to prove it worked:
    for char, variations in hiragana_dataset.items():
        print(f"Character '{char}': Found {len(variations)} hand-drawn versions.")