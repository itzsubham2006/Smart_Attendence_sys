import os
import pickle
import time
import numpy as np
import faiss
import cv2
from deepface import DeepFace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INDEX_PATH = os.path.join(PROJECT_ROOT, "embeddings", "faiss_index.bin")
LABELS_PATH = os.path.join(PROJECT_ROOT, "embeddings", "labels.pkl")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "embeddings", "threshold.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "static", "output")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
DEFAULT_THRESHOLD = 0.28

_index = None
_labels = None
_threshold = DEFAULT_THRESHOLD


def _load_index_and_labels():
    global _index, _labels, _threshold
    if _index is not None:
        return _index, _labels, _threshold
    if not os.path.exists(INDEX_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("FAISS index or labels not found. Run generate_embeddings first.")
    _index = faiss.read_index(INDEX_PATH)
    with open(LABELS_PATH, "rb") as f:
        _labels = pickle.load(f)
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "rb") as f:
            _threshold = pickle.load(f)
    if _index.ntotal != len(_labels):
        raise ValueError(f"Index ({_index.ntotal}) and labels ({len(_labels)}) mismatch.")
    return _index, _labels, _threshold


def recognize_faces(image_path):
    index, labels, threshold = _load_index_and_labels()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
        normalization="ArcFace",
    )

    present_set = set()
    results = []

    for face in faces:
        emb = np.asarray(face["embedding"], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)

        scores, indices = index.search(emb, 3)
        valid = [(s, idx) for s, idx in zip(scores[0], indices[0]) if idx >= 0 and s >= threshold]

        if not valid:
            name = "Unknown"
            status = "Unknown"
            color = (0, 0, 255)
            final_similarity = 0.0
        else:
            from collections import Counter
            label_votes = Counter()
            for s, idx in valid:
                label_votes[labels[idx]] += s
            name = label_votes.most_common(1)[0][0]
            final_similarity = max(s for s, _ in valid)
            status = "Present"
            present_set.add(name)
            color = (0, 255, 0)

        area = face.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))
        face_confidence = float(face.get("face_confidence", 0.0))

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label_text = f"{name} ({final_similarity:.2f})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(
            image, label_text,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )

        results.append({
            "face_id": len(results) + 1,
            "name": name,
            "status": status,
            "similarity": round(final_similarity, 4),
            "confidence": round(face_confidence * 100.0, 2),
            "x": x, "y": y, "w": w, "h": h,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = int(time.time())
    output_name = f"result_{ts}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Remove old result images (keep only last 5)
    old = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("result_") and f.endswith(".jpg")])
    for f in old[:-5]:
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
        except Exception:
            pass

    present_students = sorted(present_set)
    return results, present_students, f"/static/output/{output_name}"
