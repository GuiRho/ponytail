import os, random
import cv2
import matplotlib.pyplot as plt
import albumentations as A

TARGET_SIZE = 224

def aug_pipeline(shift=0.0, scale=0.0, rotate=0, brightness=0.0, contrast=0.0, p_flip=0.0, crop_scale=(0.8,1.0), crop_ratio=(0.9,1.1), dropout_holes=0, dropout_h=8, dropout_w=8, blur=0):
    transforms = [A.ShiftScaleRotate(shift_limit=shift, scale_limit=scale, rotate_limit=rotate, p=0.7), A.HorizontalFlip(p=p_flip), A.RandomResizedCrop(height=TARGET_SIZE, width=TARGET_SIZE, scale=crop_scale, ratio=crop_ratio, p=1.0), A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=0.5)]
    if dropout_holes > 0: transforms.append(A.CoarseDropout(max_holes=dropout_holes, max_height=dropout_h, max_width=dropout_w, fill_value=0, p=0.5))
    if blur > 0: transforms.append(A.Blur(blur_limit=max(3, blur|1), p=0.3))
    return A.Compose(transforms)

def aug_config():
    return {"name": "Strong Geometric + Extreme (N=2)", "num_augs_per_image": 2, "pipelines": [aug_pipeline(0.2, 0.2, 25, 0.1, 0.1, 0.7, (0.7,1.0), (0.8,1.2), 10, 10, 10, 5), aug_pipeline(0.3, 0.3, 45, 0.4, 0.4, 0.8, (0.6,1.0), (0.7,1.3), 15, 15, 15, 7)]}

def show_aug(df_row, image_dir, config):
    img = cv2.imread(os.path.join(image_dir, df_row["image"]))
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    n = config["num_augs_per_image"] + 1
    fig, axes = plt.subplots(1, n, figsize=(20,5))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    for i in range(config["num_augs_per_image"]):
        aug = random.choice(config["pipelines"])(image=img)["image"]
        axes[i+1].imshow(aug, cmap="gray"); axes[i+1].set_title(f"Aug {i+1}"); axes[i+1].axis("off")
    plt.suptitle(f"Category: {df_row.get('main_category','N/A')}"); plt.show()

if __name__ == "__main__":
    p = aug_pipeline()
    assert isinstance(p, A.Compose)
    c = aug_config()
    assert c["num_augs_per_image"] == 2
    print("augmentation OK")
