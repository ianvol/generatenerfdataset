import sys
import argparse
from llff.poses.pose_utils import gen_poses
from PIL import Image
import os
from tkinter import Tk, filedialog
from tkinter import messagebox
import numpy as np

# Create a Tkinter root window
root = Tk()
root.withdraw()

# Display a popup message
messagebox.showinfo("NeRF Dataset Generator", "Takes a set of images and generates a NeRF dataset with poses, focal length (default value used), and image shape.")

# Prompt the user to select the image files
image_files = filedialog.askopenfilenames(title="Select Image Files", filetypes=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png")))
image_filenames = sorted(image_files)

default_focal_length = 50.0  # Replace with your desired default focal length
max_image_size = (1024, 1024)  # Maximum width and height for resizing

def generate_poses(num_images):
    scenedir = args.scenedir  # Path to the scene directory
    match_type = args.match_type  # Type of matcher used
    poses = gen_poses(scenedir, match_type, num_images)
    return poses

poses = generate_poses(len(image_filenames))

images = []
heights = []
widths = []
for file in image_files:
    with open(file, "rb") as image_file:
        image = Image.open(image_file)
        # Preprocess the image as desired (e.g., resize, crop, normalize, etc.)
        image.thumbnail(max_image_size)
        images.append(np.array(image))
        heights.append(image.height)
        widths.append(image.width)

poses = np.array(poses).reshape(-1, 7)
images = np.array(images).reshape(-1, images[0].shape[0] * images[0].shape[1] * images[0].shape[2])
heights = np.array(heights).reshape(-1, 1)
widths = np.array(widths).reshape(-1, 1)

focal = np.full(len(image_filenames), default_focal_length).reshape(-1, 1)

# Prompt the user to select the save path and dataset name for the dataset
save_path = filedialog.asksaveasfilename(title="Save NeRF Dataset", defaultextension=".npz")

if save_path:
    save_path = os.path.abspath(save_path)  # Convert to absolute path to handle Unicode characters
    dataset_name = os.path.splitext(os.path.basename(save_path))[0]
    np.savez(save_path, poses=poses, images=images, focal=focal, heights=heights, widths=widths)
    messagebox.showinfo("NeRF Dataset Saved", f"NeRF dataset '{dataset_name}' saved successfully.")

else:
    messagebox.showinfo("NeRF Dataset Generation", "Dataset generation canceled.")
