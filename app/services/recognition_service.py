import os
import cv2
import pickle
import numpy as np
import faiss
from deepface import DeepFace

INDEX_PATH = "embeddings/faiss_index.bin"
LABELS_PATH = "embeddings/labels.pkl"
OUTPUT_IMAGE_PATH = "static/output/result.jpg"

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
ENFORCE_DETECTION = True
ALIGN = True
NORMALIZATION = "ArcFace"

# Tune between 0.35 and 0.45 based on your classroom data
COSINE_THRESHOLD = 0.40


def _load_index_and_labels():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Missing index: {INDEX_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

    index = faiss.read_index(INDEX_PATH)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

    if index.ntotal != len(labels):
        raise ValueError("Index and labels mismatch. Re-run embeddings generation.")

    return index, labels


def recognize_faces(image_path, threshold=COSINE_THRESHOLD):
    """
    Returns:
      results: list[dict] per-face recognition details
      present_students: list[str] unique recognized student names
      output_image_path: str path for template image src
    """
    index, labels = _load_index_and_labels()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Detect all faces + generate embeddings
    faces = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION,
        align=ALIGN,
        normalization=NORMALIZATION
    )

    present_set = set()
    results = []

    for face in faces:
        emb = np.asarray(face["embedding"], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)

        scores, indices = index.search(emb, 1)
        similarity = float(scores[0][0])
        best_idx = int(indices[0][0])

        face_confidence = float(face.get("face_confidence", 0.0))

        if best_idx >= 0 and similarity >= threshold:
            name = labels[best_idx]
            status = "Present"
            present_set.add(name)
            color = (0, 255, 0)
        else:
            name = "Unknown"
            status = "Unknown"
            color = (0, 0, 255)

        area = face.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            f"{name} {similarity:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        results.append(
            {
                "face_id": len(results) + 1,
                "name": name,
                "status": status,
                "similarity": round(similarity, 4),
                "confidence": round(face_confidence * 100.0, 2),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)

    present_students = sorted(list(present_set))
    return results, present_students, f"/{OUTPUT_IMAGE_PATH}"