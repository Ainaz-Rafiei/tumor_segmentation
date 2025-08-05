import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from unet_model import UNet

# ----------------------- Dataset -----------------------
class TumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx].replace("patient_", "segmentation_"))

        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        mask = mask / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# ----------------------- Dice Score -----------------------
def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = (pred > 0).astype(np.bool_)
    target = (target > 0).astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    return 2. * intersection / (pred.sum() + target.sum() + 1e-8)

# ----------------------- Predict -----------------------
def predict(model, img: np.ndarray) -> np.ndarray:
    model.eval()
    original_shape = img.shape[:2]
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    pred = output.squeeze().detach().numpy()
    pred = cv2.resize(pred, (original_shape[1], original_shape[0]))
    mask = (pred > 0.5).astype(np.uint8) * 255
    return np.stack([mask] * 3, axis=-1)

# ----------------------- Validation -----------------------
def validate_on_folder(model, image_dir, mask_dir):
    dice_scores = []
    tumor_positive_scores = []

    filenames = sorted(os.listdir(image_dir))
    for fname in tqdm(filenames, desc="Validating"):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace("patient_", "segmentation_"))

        img = cv2.imread(img_path)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        pred_mask = predict(model, img)
        pred_gray = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)

        score = dice_score(pred_gray, true_mask)
        dice_scores.append(score)

        if true_mask.sum() > 0:
            tumor_positive_scores.append(score)

    avg_dice = np.mean(dice_scores)
    tumor_avg_dice = np.mean(tumor_positive_scores) if tumor_positive_scores else 0.0
    tumor_sum_dice = np.sum(tumor_positive_scores) if tumor_positive_scores else 0.0

    print(f"\nAverage Dice Score on All Cases: {avg_dice:.4f}")
    print(f"Tumor-Positive Case Average: {tumor_avg_dice:.4f}")
    print(f"Tumor-Positive Case Total: {tumor_sum_dice:.2f}")
    print(f"Format for leaderboard: {tumor_sum_dice:.2f} ({tumor_avg_dice:.2f})")

    return tumor_avg_dice  # Return for saving best model

# ----------------------- Train Loop -----------------------
def train(model, train_loader, val_image_dir, val_mask_dir, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_score = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"\nEpoch {epoch} Loss: {running_loss / len(train_loader):.4f}")

        val_score = validate_on_folder(model, val_image_dir, val_mask_dir)

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved best model.")

# ----------------------- Main -----------------------
if __name__ == "__main__":
    model = UNet()
    train_dataset = TumorSegmentationDataset("data/train/imgs", "data/train/labels")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    train(model, train_loader, "data/val/imgs", "data/val/labels", epochs=20, lr=1e-3)
# Format for leaderboard: 48.06 (0.26)