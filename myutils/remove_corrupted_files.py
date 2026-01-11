import os
import argparse
from PIL import Image

def check_images(image_dir, delete_corrupt=False):
    # Ensure the path exists
    if not os.path.exists(image_dir):
        print(f"Error: The directory '{image_dir}' does not exist.")
        return

    # Get list of files
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Checking {len(files)} images in: {image_dir}")
    corrupt_count = 0

    for filename in files:
        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                # verify() catches many corruption issues but doesn't decode the whole image
                img.verify()
        except Exception:
            corrupt_count += 1
            print(f"\n[CORRUPT] {filepath}")
            if delete_corrupt:
                os.remove(filepath)
                print(f"Successfully deleted: {filename}")

    print(f"\nScan complete.")
    print(f"Total images checked: {len(files)}")
    print(f"Corrupt files found: {corrupt_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan a directory for corrupt image files.")

    # Positional argument: the directory path
    parser.add_argument("path", help="Path to the directory containing images.")

    # Optional flag: --delete
    parser.add_argument("--delete", action="store_true", help="Delete the corrupt files if found.")

    args = parser.parse_args()

    check_images(args.path, args.delete)