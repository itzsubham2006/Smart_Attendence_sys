import numpy as np
import faiss
import os
from deepface import DeepFace

EMBEDDINGS_PATH = "embeddings"

index_path = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    print("⚠️ FAISS index not found. Please train model first.")
    index = None
labels = np.load(os.path.join(EMBEDDINGS_PATH, "labels.npy"))


THRESHOLD = 0.4

def recognize_faces(image_path):
    results = []

    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="retinaface"
    )

    for face in faces:
        try:
            if face["confidence"] < 0.95:
                continue

            embedding = DeepFace.represent(
                img_path=face["face"],
                model_name="ArcFace",
                enforce_detection=False
            )[0]["embedding"]

            query = np.array([embedding])

            D, I = index.search(query, k=1)

            distance = D[0][0]
            idx = I[0][0]

            if distance < THRESHOLD:
                label = labels[idx]
                name, roll = label.split("_")

                confidence = 1 - distance

                results.append({
                    "name": name,
                    "roll": roll,
                    "confidence": float(confidence)
                })

        except Exception as e:
            print("Error:", e)

    unique = {(r["roll"]): r for r in results}
    return list(unique.values())