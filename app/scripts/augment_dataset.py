import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")


def augment_image(image):
    augmented = []
    h, w = image.shape[:2]

    
    augmented.append(image)

  
    augmented.append(cv2.flip(image, 1))

   
    for factor in [0.85, 1.15]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

   
    for alpha in [0.9, 1.1]:
        augmented.append(cv2.convertScaleAbs(image, alpha=alpha, beta=0))

    return augmented


def process_dataset():
    total = 0
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not images:
            continue

        print(f"Processing {person} ({len(images)} images)...")
        count = 0

        for img_name in images:
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            variants = augment_image(image)
            for aug_img in variants[1:]:
                aug_name = f"aug_{count}_{img_name}"
                save_path = os.path.join(person_path, aug_name)
                cv2.imwrite(save_path, aug_img)
                count += 1

        total += count
        print(f"  Added {count} augmented images for {person}")

    print(f"\nTotal augmented images created: {total}")


if __name__ == "__main__":
    process_dataset()
