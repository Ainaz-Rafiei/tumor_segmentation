import os
import shutil
from sklearn.model_selection import train_test_split

img_dir = "data/patients/imgs"
mask_dir = "data/patients/labels"

train_img_dir = "data/train/imgs"
train_mask_dir = "data/train/labels"
val_img_dir = "data/val/imgs"
val_mask_dir = "data/val/labels"

# Make folders
for folder in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
    os.makedirs(folder, exist_ok=True)

# Get filenames
all_images = sorted(os.listdir(img_dir))
train_imgs, val_imgs = train_test_split(all_images, test_size=0.1, random_state=42)

# Copy files
for img_name in train_imgs:
    shutil.copy(os.path.join(img_dir, img_name), os.path.join(train_img_dir, img_name))
    mask_name = img_name.replace("patient_", "segmentation_")
    shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(train_mask_dir, mask_name))

for img_name in val_imgs:
    shutil.copy(os.path.join(img_dir, img_name), os.path.join(val_img_dir, img_name))
    mask_name = img_name.replace("patient_", "segmentation_")
    shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(val_mask_dir, mask_name))

print("✅ Train/validation split done.")
