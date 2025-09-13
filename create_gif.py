from PIL import Image
import os

# Path to the folder containing images
image_folder = "/home/AutoDP/theyanesh.er/flownav/logs/flownav/flownav_2025_09_12_20_06_33/visualize/test/epoch0/action_sampling_prediction"
output_gif = "output.gif"

# Get all image file names sorted (important for correct frame order)
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Open images and store them in a list
images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

# Save as GIF
# duration is the time between frames in milliseconds
# loop=0 means infinite loop
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=200,
    loop=0
)

print(f"GIF saved as {output_gif}")
