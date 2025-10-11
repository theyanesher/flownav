from torchvision import transforms
from torch.utils.data import DataLoader
from nuscenes_dataloader import NuScenesTemporalDataset

transform = transforms.Compose([
    transforms.Resize((224, 400)),
    transforms.ToTensor()
])

dataset = NuScenesTemporalDataset(
    nusc_root='/share1/ad_dataset/nuscenes',
    version='v1.0-mini',
    camera='CAM_FRONT',
    context_len=5,
    future_len=8,
    transform=transform,
    include_goal_image=True
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

print(f"Dataset size: {len(dataset)}")

batch = next(iter(dataloader))
# breakpoint()
print('odom:', batch['future_odom'])  # [B, future_len, 3]
print('odom shape:', batch['future_odom'].shape)  # [B, future_len, 3]
# breakpoint()
print("current:", batch['current_img'].shape)     # [B, 3, 224, 224]
print("context:", batch['context_imgs'].shape)    # [B, 5, 3, 224, 224]
print("future odom:", batch['future_odom'].shape) # [B, 8, 3]
print("goal image:", batch['goal_img'][0].shape)  # [3, 224, 224]

import os
from torchvision.utils import save_image

# Create output directory
output_dir = "./nuscenes_images"
os.makedirs(output_dir, exist_ok=True)

# Batch size
B = batch['current_img'].shape[0]

for b in range(B):
    # --- Save current image ---
    # current_img = batch['current_img'][b]  # [3, H, W]
    # save_image(current_img, os.path.join(output_dir, f"batch{b}_current.png"))

    # --- Save context images ---
    context_imgs = batch['context_imgs'][b]  # [context_len, 3, H, W]
    context_len = context_imgs.shape[0]
    for i in range(context_len):
        img = context_imgs[i]
        save_image(img, os.path.join(output_dir, f"batch{b}_context_{i}.png"))

    # --- Save goal image if exists ---
    goal_img = batch['goal_img'][b]  # [3, H, W] or None
    if goal_img is not None:
        save_image(goal_img, os.path.join(output_dir, f"batch{b}_goal.png"))

print(f"Saved all images to {output_dir}")
