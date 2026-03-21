import numpy as np
import faiss
import os
from deepface import DeepFace

EMBEDDINGS_PATH = "embeddings"

index_path = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")
labels_path = os.path.join(EMBEDDINGS_PATH, "labels.npy")

if os.path.exists(index_path) and os.path.exists(labels_path):
    index = faiss.read_index(index_path)
    labels = np.load(labels_path)
else:
    print(" Model not trained yet. Run embedding script first.")
    index = None
    labels = None

THRESHOLD = 0.4


def recognize_faces(image_path):
    if index is None or labels is None:
        print("Recognition system not ready.")
        return []

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
                img_path=None,
                img=face["face"],
                model_name="ArcFace",
                enforce_detection=False
            )[0]["embedding"]

            query = np.array([embedding]).astype("float32")

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