import os
import numpy as np
from deepface import DeepFace

DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings"

def generate_embeddings():
    embeddings = []
    labels = []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="retinaface"
                )

                embedding = reps[0]["embedding"]

                embeddings.append(embedding)
                labels.append(person)

                print(f"Processed: {img_path}")

            except Exception as e:
                print(f"Error: {img_path} → {e}")

    np.save(os.path.join(EMBEDDINGS_PATH, "embeddings.npy"), np.array(embeddings))
    np.save(os.path.join(EMBEDDINGS_PATH, "labels.npy"), np.array(labels))

    print("✅ Embeddings saved!")