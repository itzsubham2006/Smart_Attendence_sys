import os
import cv2
import numpy as np

DATASET_PATH = "dataset"
AUGMENTED_COUNT = 15  

def augment_image(image):
    augmented_images = []

    h, w = image.shape[:2]

    for i in range(AUGMENTED_COUNT):
        img = image.copy()

       
        angle = np.random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

       
        value = np.random.randint(-30, 30)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

       
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

     
        zoom = np.random.uniform(0.9, 1.1)
        center_x, center_y = w // 2, h // 2
        new_w, new_h = int(w * zoom), int(h * zoom)

        x1 = max(center_x - new_w // 2, 0)
        y1 = max(center_y - new_h // 2, 0)
        x2 = min(center_x + new_w // 2, w)
        y2 = min(center_y + new_h // 2, h)

        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (w, h))

       
        if np.random.rand() > 0.7:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        augmented_images.append(img)

    return augmented_images


def process_dataset():
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        images = os.listdir(person_path)

        print(f"Processing {person}...")

        count = 0

        for img_name in images:
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            augmented_images = augment_image(image)

            for aug_img in augmented_images:
                new_name = f"aug_{count}.jpg"
                save_path = os.path.join(person_path, new_name)

                cv2.imwrite(save_path, aug_img)
                count += 1

        print(f"{person} now has augmented images!")

if __name__ == "__main__":
    process_dataset()
    
