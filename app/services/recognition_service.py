import os
import cv2
import numpy as np
import pickle
import faiss
from deepface import DeepFace

EMBEDDINGS_PATH = "embeddings"
OUTPUT_PATH = "static/output"
THRESHOLD = 0.5

os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_model():
    index = faiss.read_index(os.path.join(EMBEDDINGS_PATH, "faiss_index.bin"))

    with open(os.path.join(EMBEDDINGS_PATH, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)

    return index, labels


def distance_to_confidence(distance):
    return round(max(0, (1 - distance)) * 100, 2)


def recognize_faces(image_path):
    index, labels = load_model()

    img = cv2.imread(image_path)

    if img is None:
        return [], None

    results = []
    seen = set()

    try:
        detections = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=False
        )

        print(f"Detected {len(detections)} faces")

        for i, face_data in enumerate(detections):
            face = face_data["face"]
            region = face_data["facial_area"]

            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            face = (face * 255).astype("uint8")

            try:
                embedding = DeepFace.represent(
                    img_path=face,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                embedding = np.array(embedding).astype("float32")
                faiss.normalize_L2(embedding.reshape(1, -1))

                D, I = index.search(embedding.reshape(1, -1), k=1)

                distance = D[0][0]
                idx = I[0][0]

                if distance < THRESHOLD:
                    full_name = labels[idx]

                    parts = full_name.split("_")
                    student_name = parts[0]
                    roll = parts[1] if len(parts) > 1 else "N/A"

                    confidence = distance_to_confidence(distance)

                   
                    if roll in seen:
                        continue
                    seen.add(roll)

                    color = (0, 255, 0)  
                    label = f"{student_name} ({confidence}%)"

                else:
                    student_name = "Unknown"
                    roll = "-"
                    confidence = 0

                    color = (0, 0, 255)  
                    label = "Unknown"

            
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                results.append({
                    "name": student_name,
                    "roll": roll,
                    "confidence": confidence,
                    "face_id": i
                })

            except Exception as e:
                print(f"Error processing face {i}: {e}")

    except Exception as e:
        print("Face detection failed:", e)

    
    output_file = os.path.join(OUTPUT_PATH, "result.jpg")
    cv2.imwrite(output_file, img)

    return results, output_file