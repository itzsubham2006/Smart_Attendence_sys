import os
import pickle
import numpy as np
import faiss
from deepface import DeepFace

DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings"

INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")
LABELS_PATH = os.path.join(EMBEDDINGS_PATH, "labels.pkl")
CACHE_PATH = os.path.join(EMBEDDINGS_PATH, "embeddings_cache.pkl")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"  # use "opencv" for more speed, less accuracy
ENFORCE_DETECTION = True
ALIGN = True
NORMALIZATION = "ArcFace"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in VALID_EXTENSIONS


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_data):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f)


def pick_best_face(face_results):
    """If multiple faces found in one image, use largest face."""
    if not face_results:
        return None

    if len(face_results) == 1:
        return face_results[0]["embedding"]

    def area(face):
        fa = face.get("facial_area", {})
        return fa.get("w", 0) * fa.get("h", 0)

    best = max(face_results, key=area)
    return best["embedding"]


def generate_embeddings():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset folder not found: {DATASET_PATH}")
        return

    print("Generating embeddings...")

    old_cache = load_cache()
    new_cache = {}

    embeddings = []
    labels = []

    total_images = 0
    cache_hits = 0
    processed = 0
    errors = 0

    for person in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in sorted(os.listdir(person_path)):
            if not is_image_file(img_name):
                continue

            total_images += 1
            img_path = os.path.join(person_path, img_name)
            cache_key = os.path.abspath(img_path)

            try:
                mtime = os.path.getmtime(img_path)
                cached = old_cache.get(cache_key)

                if (
                    cached
                    and cached.get("mtime") == mtime
                    and cached.get("label") == person
                ):
                    embeddings.append(cached["embedding"])
                    labels.append(person)
                    new_cache[cache_key] = cached
                    cache_hits += 1
                    continue

                face_results = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=ENFORCE_DETECTION,
                    align=ALIGN,
                    normalization=NORMALIZATION
                )

                emb = pick_best_face(face_results)
                if emb is None:
                    raise ValueError("No embedding generated")

                embeddings.append(emb)
                labels.append(person)

                new_cache[cache_key] = {
                    "mtime": mtime,
                    "label": person,
                    "embedding": emb
                }

                processed += 1
                print(f"Processed: {person}/{img_name}")

            except Exception as e:
                errors += 1
                print(f"Error: {person}/{img_name} -> {e}")

    if len(embeddings) == 0:
        print("No embeddings created. Check dataset images.")
        return

    emb_np = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb_np)

    dim = emb_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
    index.add(emb_np)

    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    save_cache(new_cache)

    print("\nEmbeddings and FAISS index created successfully.")
    print(f"Total images : {total_images}")
    print(f"Processed    : {processed}")
    print(f"Cache hits   : {cache_hits}")
    print(f"Errors       : {errors}")
    print(f"Embeddings   : {len(labels)}")


if __name__ == "__main__":
    generate_embeddings()