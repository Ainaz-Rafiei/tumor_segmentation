import numpy as np
import torch
import numpy as np
import cv2
from unet_model import UNet
import os
from tqdm import tqdm

# Load model (You can later replace this with a pre-trained one)
model = UNet()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
# model = UNet()
# model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
# model.eval()
def preprocess(img: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img, (128, 128))  # Resize for simplicity
    img = img / 255.0  # Normalize
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def postprocess(pred: torch.Tensor, original_shape) -> np.ndarray:
    pred = pred.squeeze().detach().numpy()
    pred = cv2.resize(pred, (original_shape[1], original_shape[0]))
    mask = (pred > 0.5).astype(np.uint8) * 255
    return np.stack([mask] * 3, axis=-1)  # 3-channel binary mask

def predict(img: np.ndarray) -> np.ndarray:
    original_shape = img.shape[:2]
    input_tensor = preprocess(img)
    with torch.no_grad():
        output = model(input_tensor)
    return postprocess(output, original_shape)

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = (pred > 0).astype(np.bool_)
    target = (target > 0).astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    return 2. * intersection / (pred.sum() + target.sum() + 1e-8)

def load_image(path):
    return cv2.imread(path)

def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# def validate_on_folder(image_dir, mask_dir):
#     dice_scores = []

#     filenames = sorted(os.listdir(image_dir))
#     for fname in tqdm(filenames, desc="Validating"):
#         img_path = os.path.join(image_dir, fname)
#         mask_fname = fname.replace("patient_", "segmentation_")
#         mask_path = os.path.join(mask_dir, mask_fname)

#         img = load_image(img_path)
#         true_mask = load_mask(mask_path)

#         pred_mask = predict(img)
#         pred_gray = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)

#         score = dice_score(pred_gray, true_mask)
#         dice_scores.append(score)

#     print(f"\n Average Dice Score on Validation Set: {np.mean(dice_scores):.4f}")
def validate_on_folder(image_dir, mask_dir):
    dice_scores = []
    tumor_positive_scores = []

    filenames = sorted(os.listdir(image_dir))
    for fname in tqdm(filenames, desc="Validating"):
        img_path = os.path.join(image_dir, fname)
        mask_fname = fname.replace("patient_", "segmentation_")
        mask_path = os.path.join(mask_dir, mask_fname)

        img = load_image(img_path)
        true_mask = load_mask(mask_path)

        pred_mask = predict(img)
        pred_gray = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)

        score = dice_score(pred_gray, true_mask)
        dice_scores.append(score)

        if true_mask.sum() > 0:  # Only include tumor-positive cases
            tumor_positive_scores.append(score)

    # Overall average Dice
    avg_dice = np.mean(dice_scores)

    # Tumor-positive cases only
    if tumor_positive_scores:
        tumor_avg_dice = np.mean(tumor_positive_scores)
        tumor_sum_dice = np.sum(tumor_positive_scores)
    else:
        tumor_avg_dice = 0.0
        tumor_sum_dice = 0.0

    print(f"\nAverage Dice Score on All Cases: {avg_dice:.4f}")
    print(f"Tumor-Positive Case Average: {tumor_avg_dice:.4f}")
    print(f"Tumor-Positive Case Total: {tumor_sum_dice:.2f}")
    print(f"Format for leaderboard: {tumor_sum_dice:.2f} ({tumor_avg_dice:.2f})")

# Run it
if __name__ == "__main__":
    validate_on_folder("data/val/imgs", "data/val/labels")

### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
# def predict(img: np.ndarray) -> np.ndarray:
#     threshold = 50
#     segmentation = get_threshold_segmentation(img,threshold)
#     return segmentation
### DUMMY MODEL ###
def get_threshold_segmentation(img:np.ndarray, threshold:int) -> np.ndarray:
    return (img < threshold).astype(np.uint8)*255
