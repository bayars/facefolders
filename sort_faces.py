#!/usr/bin/env python3
"""Sort JPEG photos into per-person folders using InsightFace embeddings + HDBSCAN."""
import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

EXTS = {".jpg", ".jpeg"}
CACHE_NPZ = "faces.npz"
CACHE_JSON = "faces.json"
MARKER = ".person.json"  # written into every person folder; survives renaming the folder
OUT_DIRS_FIXED = ("unknown", "no_faces")

# insightface triggers a scikit-image deprecation warning on every face
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="folder with photos (scanned recursively)")
    ap.add_argument("--output", type=Path, required=True, help="folder for sorted output")
    ap.add_argument("--min-cluster-size", type=int, default=3, help="min faces to form a person")
    ap.add_argument("--min-samples", type=int, default=2, help="HDBSCAN min_samples (lower = fewer noise faces)")
    ap.add_argument("--method", choices=["eom", "leaf"], default="eom", help="HDBSCAN cluster selection")
    ap.add_argument("--merge-thresh", type=float, default=0.5,
                    help="merge clusters / match named people if centroid cosine similarity >= this (1 = off)")
    ap.add_argument("--det-thresh", type=float, default=0.6, help="min face detection score")
    ap.add_argument("--min-face", type=int, default=40, help="min face size in px (on the resized image)")
    ap.add_argument("--max-side", type=int, default=1600, help="resize photos so longest side <= this")
    ap.add_argument("--det-size", type=int, default=640, help="detector input size")
    ap.add_argument("--workers", type=int, default=4, help="threads decoding photos ahead of the GPU")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--recluster", action="store_true", help="reuse cached embeddings, skip detection")
    return ap.parse_args()


def list_photos(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def load_image(path, max_side=None):
    """Return (BGR array with EXIF rotation applied, scale factor vs. original)."""
    with Image.open(path) as im:
        orig = max(im.size)
        if max_side and orig > max_side:
            # For JPEGs, thumbnail() decodes at 1/2, 1/4 or 1/8 size via draft mode,
            # which is much faster than a full decode followed by a resize.
            im.thumbnail((max_side, max_side), reducing_gap=1.0)
        scale = max(im.size) / orig
        im = ImageOps.exif_transpose(im).convert("RGB")
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1]), scale


def iter_images(paths, max_side, workers):
    """Yield (path, image, scale, error) in order, decoding up to 2*workers photos ahead."""
    def job(p):
        try:
            return p, *load_image(p, max_side), None
        except Exception as e:
            return p, None, None, str(e)

    with ThreadPoolExecutor(workers) as ex:
        it = iter(paths)
        pending = deque(ex.submit(job, p) for _, p in zip(range(workers * 2), it))
        while pending:
            done = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(job, nxt))
            yield done


def build_app(det_size, force_cpu):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    model_root = os.environ.get("INSIGHTFACE_ROOT", "~/.insightface")
    available = ort.get_available_providers()

    def make(use_gpu):
        providers = (["CUDAExecutionProvider"] if use_gpu else []) + ["CPUExecutionProvider"]
        with contextlib.redirect_stdout(io.StringIO()):  # insightface prints a line per model
            app = FaceAnalysis(name="buffalo_l", root=model_root, providers=providers,
                               allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
        app.get(np.zeros((det_size, det_size, 3), dtype=np.uint8))  # smoke test kernels
        return app

    # Without a GPU device ORT's CUDA provider can segfault instead of raising, so check first.
    has_device = os.path.exists("/dev/nvidiactl")
    if not force_cpu and not has_device:
        print("!! No NVIDIA device visible (did you pass --gpus all?)", file=sys.stderr)
    use_gpu = not force_cpu and has_device and "CUDAExecutionProvider" in available
    try:
        app = make(use_gpu)
    except Exception as e:
        if not use_gpu:
            raise
        print(f"!! GPU init failed ({e}); falling back to CPU", file=sys.stderr)
        app = make(False)

    used = app.models["detection"].session.get_providers()
    if "CUDAExecutionProvider" in used:
        print(f"Using GPU (onnxruntime {ort.__version__})")
    else:
        print("!! Running on CPU - this will be slower", file=sys.stderr)
    return app


def detect(photos, input_root, args):
    app = build_app(args.det_size, args.cpu)
    index = {p: i for i, p in enumerate(photos)}
    faces, embs, failed = [], [], {}
    images = iter_images(photos, args.max_side, args.workers)
    for path, img, scale, err in tqdm(images, total=len(photos), desc="Detecting", unit="photo"):
        i = index[path]
        if err:
            failed[i] = err
            continue
        for f in app.get(img):
            x1, y1, x2, y2 = f.bbox
            if f.det_score < args.det_thresh or min(x2 - x1, y2 - y1) < args.min_face:
                continue
            faces.append({
                "photo": i,
                "bbox": (f.bbox / scale).round(1).tolist(),  # original-image coordinates
                "score": round(float(f.det_score), 4),
            })
            embs.append(f.normed_embedding)
    embs = np.asarray(embs, dtype=np.float32).reshape(-1, 512)
    meta = {
        "input": str(input_root),
        "photos": [str(p.relative_to(input_root)) for p in photos],
        "failed": {str(k): v for k, v in failed.items()},
        "faces": faces,
    }
    return meta, embs


def save_cache(out, meta, embs):
    np.savez_compressed(out / CACHE_NPZ, embeddings=embs)
    (out / CACHE_JSON).write_text(json.dumps(meta))


def load_cache(out):
    meta = json.loads((out / CACHE_JSON).read_text())
    embs = np.load(out / CACHE_NPZ)["embeddings"]
    return meta, embs


def cluster(embs, args):
    if len(embs) < max(args.min_cluster_size, 2):
        return np.full(len(embs), -1)
    import hdbscan

    # Embeddings are L2-normalised, so euclidean distance is monotonic with cosine distance.
    return hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.method,
    ).fit_predict(embs.astype(np.float64))


def centroid(vecs):
    c = vecs.mean(axis=0)
    return c / np.linalg.norm(c)


def merge_clusters(embs, labels, thresh):
    """Repeatedly merge the two clusters with the most similar centroids while similarity >= thresh.

    HDBSCAN often splits one person into several clusters (glasses, age, lighting).
    """
    labels = labels.copy()
    while True:
        ids = sorted(set(labels) - {-1})
        if len(ids) < 2:
            return labels
        cents = np.stack([centroid(embs[labels == c]) for c in ids])
        sim = cents @ cents.T
        np.fill_diagonal(sim, -1)
        a, b = np.unravel_index(sim.argmax(), sim.shape)
        if sim[a, b] < thresh:
            return labels
        print(f"  merging clusters (similarity {sim[a, b]:.2f})")
        labels[labels == ids[b]] = ids[a]


def load_known_people(out):
    """Folders from a previous run, keyed by their current (possibly user-renamed) name."""
    known = {}
    for d in out.iterdir():
        marker = d / MARKER
        if d.is_dir() and marker.exists():
            try:
                known[d.name] = np.asarray(json.loads(marker.read_text())["centroid"], dtype=np.float32)
            except (ValueError, KeyError):
                pass
    return known


def name_clusters(embs, labels, known, thresh):
    """Map cluster id -> folder name, reusing names of matching folders from earlier runs.

    New people get person_NNN numbers not used before; order is largest cluster first.
    """
    ids = [c for c, _ in Counter(l for l in labels if l >= 0).most_common()]
    cents = {c: centroid(embs[labels == c]) for c in ids}
    names = {}
    if known and ids:
        known_names = list(known)
        sim = np.stack([cents[c] for c in ids]) @ np.stack([known[n] for n in known_names]).T
        # Greedy one-to-one matching, best pairs first.
        for flat in np.argsort(sim, axis=None)[::-1]:
            ci, ki = np.unravel_index(flat, sim.shape)
            if sim[ci, ki] < thresh:
                break
            c, n = ids[ci], known_names[ki]
            if c not in names and n not in names.values():
                names[c] = n
    used = set(names.values()) | set(known)
    width = max(3, len(str(len(ids) + len(known))))
    num = 1
    for c in ids:
        if c in names:
            continue
        while f"person_{num:0{width}d}" in used:
            num += 1
        names[c] = f"person_{num:0{width}d}"
        used.add(names[c])
    return names, cents


def flat_name(rel):
    return rel.replace(os.sep, "__")


def save_face_crop(src, bbox, dst, margin=0.3):
    img, _ = load_image(src)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    Image.fromarray(img[y1:y2, x1:x2, ::-1]).save(dst, quality=92)


def clear_previous_output(out):
    for d in out.iterdir():
        if d.is_dir() and ((d / MARKER).exists() or d.name.startswith("person_") or d.name in OUT_DIRS_FIXED):
            shutil.rmtree(d)


def check_writable(out):
    """Fail fast instead of after the (slow) detection pass."""
    probe = out / ".write_test"
    try:
        probe.write_text("")
        probe.unlink()
    except OSError as e:
        st = out.stat()
        sys.exit(
            f"Cannot write to output dir ({e}).\n"
            f"  dir owner uid:gid = {st.st_uid}:{st.st_gid}, container runs as {os.getuid()}:{os.getgid()}\n"
            f"  Fix on the host: sudo chown -R $(id -u):$(id -g) <OUTPUT_DIR>, "
            f"and set UID/GID in .env to the output of `id -u` / `id -g`."
        )


def write_output(meta, labels, names, cents, out):
    input_root = Path(meta["input"])
    photos = meta["photos"]
    failed = {int(k) for k in meta["failed"]}

    faces_by_photo = defaultdict(list)
    best_face = {}  # cluster id -> face index with highest det score
    for fi, (face, cid) in enumerate(zip(meta["faces"], labels)):
        faces_by_photo[face["photo"]].append(cid)
        if cid >= 0 and (cid not in best_face or face["score"] > meta["faces"][best_face[cid]]["score"]):
            best_face[cid] = fi

    clear_previous_output(out)
    for cid, name in names.items():
        (out / name).mkdir()
        (out / name / MARKER).write_text(json.dumps({"centroid": cents[cid].round(6).tolist()}))

    stats = Counter()
    rows = []
    for i, rel in enumerate(tqdm(photos, desc="Copying", unit="photo")):
        src = input_root / rel
        cids = faces_by_photo.get(i, [])
        people = sorted({names[c] for c in cids if c >= 0})
        if i in failed:
            dests, category = [], "failed"
        elif people:
            dests, category = people, "people"
        elif cids:
            dests, category = ["unknown"], "unknown"
        else:
            dests, category = ["no_faces"], "no_faces"
        for d in dests:
            (out / d).mkdir(exist_ok=True)
            shutil.copy2(src, out / d / flat_name(rel))
        stats[category] += 1
        rows.append([rel, len(cids), ";".join(people), category])

    for cid, fi in best_face.items():
        face = meta["faces"][fi]
        save_face_crop(input_root / photos[face["photo"]], face["bbox"], out / names[cid] / "_face.jpg")

    with open(out / "report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "n_faces", "persons", "category"])
        w.writerows(rows)

    n_noise = int((labels == -1).sum())
    n_named = sum(not n.startswith("person_") for n in names.values())
    print("\n=== Summary ===")
    print(f"Photos:            {len(photos)}")
    print(f"  with known people: {stats['people']}")
    print(f"  faces unclustered: {stats['unknown']}  -> unknown/")
    print(f"  no faces:          {stats['no_faces']}  -> no_faces/")
    print(f"  failed to read:    {stats['failed']}")
    print(f"Faces detected:    {len(labels)}  (noise: {n_noise})")
    print(f"People (clusters): {len(names)}  ({n_named} named)")
    print(f"Report:            {out / 'report.csv'}")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check_writable(args.output)

    if args.recluster:
        if not (args.output / CACHE_JSON).exists():
            sys.exit(f"No cache in {args.output}; run once without --recluster first.")
        meta, embs = load_cache(args.output)
        print(f"Loaded {len(embs)} cached faces from {len(meta['photos'])} photos")
    else:
        photos = list_photos(args.input)
        if not photos:
            sys.exit(f"No JPEG files found in {args.input}")
        print(f"Found {len(photos)} photos")
        meta, embs = detect(photos, args.input, args)
        save_cache(args.output, meta, embs)

    labels = merge_clusters(embs, cluster(embs, args), args.merge_thresh)
    names, cents = name_clusters(embs, labels, load_known_people(args.output), args.merge_thresh)
    write_output(meta, labels, names, cents, args.output)


if __name__ == "__main__":
    main()
