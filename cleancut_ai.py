import time
from rembg import remove
from PIL import Image

def remove_background(input_path: str, output_path: str):
    print(f"Reading local image: {input_path}...")
    try:
        # 1. Start the benchmark timer
        start_time = time.perf_counter()

        # 2. Open the image using Pillow
        input_image = Image.open(input_path)
        
        print("Processing image on local CPU... (This might take a few seconds)")
        
        # 3. Run the U^2-Net background removal AI model
        output_image = remove(input_image)
        
        # 4. Save as PNG to preserve the transparent background
        output_image.save(output_path, format='PNG')
        
        # 5. Stop the timer and calculate elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        print(f"Success! Saved result to {output_path}")
        print(f"⏱️ Benchmark: Total processing time was {elapsed_time:.2f} seconds.")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{input_path}'. Please make sure the file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define your input and output filenames
    input_filename = "backpack.png"
    output_filename = "nobg_backpack.png"
    
    # Run the function
    remove_background(input_filename, output_filename)