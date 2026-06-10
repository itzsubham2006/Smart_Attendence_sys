import os
import pickle
import time
import numpy as np
import faiss
from deepface import DeepFace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "embeddings")

INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")
LABELS_PATH = os.path.join(EMBEDDINGS_PATH, "labels.pkl")
CACHE_PATH = os.path.join(EMBEDDINGS_PATH, "embeddings_cache.pkl")
CENTROIDS_PATH = os.path.join(EMBEDDINGS_PATH, "centroids.pkl")
THRESHOLD_PATH = os.path.join(EMBEDDINGS_PATH, "threshold.pkl")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif", ".avif"}

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in VALID_EXTENSIONS


def pick_largest_face_embedding(face_results):
    if not face_results:
        return None
    if len(face_results) == 1:
        return face_results[0]["embedding"]

    def area(face):
        fa = face.get("facial_area", {})
        return fa.get("w", 0) * fa.get("h", 0)

    best = max(face_results, key=area)
    return best["embedding"]


def compute_centroids_by_label(embeddings_by_label):
    centroids = []
    centroid_labels = []
    for label, embs in embeddings_by_label.items():
        if len(embs) == 0:
            continue
        arr = np.array(embs, dtype=np.float32)
        centroid = arr.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)
        centroid_labels.append(label)
    return np.array(centroids, dtype=np.float32), centroid_labels


def compute_optimal_threshold(embeddings_by_label, centroids_np, centroid_labels):
    label_to_centroid = dict(zip(centroid_labels, centroids_np))
    intra_sims = []
    inter_sims = []

    for label, embs in embeddings_by_label.items():
        centroid = label_to_centroid.get(label)
        if centroid is None:
            continue
        for emb in embs:
            emb_norm = emb / np.linalg.norm(emb)
            sim = float(np.dot(emb_norm, centroid))
            intra_sims.append(sim)

    for i in range(len(centroid_labels)):
        for j in range(i + 1, len(centroid_labels)):
            sim = float(np.dot(centroids_np[i], centroids_np[j]))
            inter_sims.append(sim)

    if not intra_sims:
        return 0.40

    intra_mean = float(np.mean(intra_sims))
    intra_std = float(np.std(intra_sims))
    threshold = max(0.25, intra_mean - 2.5 * intra_std)
    return round(threshold, 4)


def generate_embeddings():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset folder not found: {DATASET_PATH}")
        return

    print("=" * 60)
    print("GENERATING EMBEDDINGS & BUILDING FAISS INDEX")
    print("=" * 60)

    old_cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                old_cache = pickle.load(f)
        except Exception:
            pass

    new_cache = {}
    embeddings_by_label = {}

    total_images = 0
    processed = 0
    cache_hits = 0
    errors = 0

    for person in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        embeddings_by_label[person] = []

        for img_name in sorted(os.listdir(person_path)):
            if not is_image_file(img_name):
                continue
            total_images += 1
            img_path = os.path.join(person_path, img_name)
            cache_key = os.path.abspath(img_path)

            try:
                mtime = os.path.getmtime(img_path)
                cached = old_cache.get(cache_key)
                if cached and cached.get("mtime") == mtime and cached.get("label") == person:
                    for emb in cached["embeddings"]:
                        embeddings_by_label[person].append(np.array(emb))
                    new_cache[cache_key] = cached
                    cache_hits += 1
                    continue

                faces = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                    normalization="ArcFace",
                )

                person_embs = []
                for face in faces:
                    emb = face.get("embedding")
                    if emb is not None:
                        person_embs.append(np.array(emb))

                if not person_embs:
                    errors += 1
                    continue

                for emb in person_embs:
                    embeddings_by_label[person].append(emb)

                new_cache[cache_key] = {
                    "mtime": mtime,
                    "label": person,
                    "embeddings": [e.tolist() for e in person_embs],
                }
                processed += 1
                print(f"  {person} / {img_name} -> {len(person_embs)} embeddings")

            except Exception as e:
                errors += 1
                print(f"  ERROR {person}/{img_name}: {e}")

    non_empty = {k: v for k, v in embeddings_by_label.items() if len(v) > 0}
    if len(non_empty) == 0:
        print("No embeddings created.")
        return

    centroids_np, centroid_labels = compute_centroids_by_label(non_empty)
    faiss.normalize_L2(centroids_np)

    dim = centroids_np.shape[1]
    n_people = len(centroid_labels)

    if n_people >= 10:
        n_centroids = max(1, min(n_people, n_people // 2))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
        index.train(centroids_np)
        index.add(centroids_np)
        index.nprobe = min(n_centroids, 3)
        index_type = f"IVFFlat ({n_centroids} centroids, nprobe={index.nprobe})"
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(centroids_np)
        index_type = "FlatIP (brute force)"

    threshold = compute_optimal_threshold(non_empty, centroids_np, centroid_labels)

    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(centroid_labels, f)
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(threshold, f)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(new_cache, f)

    print(f"\nTotal images  : {total_images}")
    print(f"Processed     : {processed}")
    print(f"Cache hits    : {cache_hits}")
    print(f"Errors        : {errors}")
    print(f"People        : {len(centroid_labels)}")
    print(f"Threshold     : {threshold}")
    print(f"Index type    : {index_type}")
    print("Embeddings generated successfully.")


if __name__ == "__main__":
    start = time.time()
    generate_embeddings()
    print(f"Total time: {time.time() - start:.1f}s")
