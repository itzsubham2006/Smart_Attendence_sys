import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
import faiss

DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings"

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)


def generate_embeddings():
    embeddings = []
    labels = []

    print("Generating embeddings...")

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",  
                    enforce_detection=True
                )[0]["embedding"]

                embeddings.append(embedding)
                labels.append(person)

                print(f"Processed: {img_name}")

            except Exception as e:
                print(f"Error with {img_name}: {e}")

    if len(embeddings) == 0:
        print("No embeddings created. Check dataset!")
        return

    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)

  
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

   
    faiss.write_index(index, os.path.join(EMBEDDINGS_PATH, "faiss_index.bin"))

    with open(os.path.join(EMBEDDINGS_PATH, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

    print("Embeddings and FAISS index created successfully!")



if __name__ == "__main__":
    generate_embeddings()