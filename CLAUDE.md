# facefolders

Sorts JPEG photos into per-person folders: InsightFace `buffalo_l` (SCRFD detection + ArcFace
512-d embeddings) → HDBSCAN clustering → centroid merge → copy photos into `person_NNN/` folders.
Runs in Docker on a home server with a **Tesla P4 (Pascal, sm_61, 8 GB)** that also runs Ollama.

## Files
- `sort_faces.py`: the whole pipeline (single file, intentionally)
- `Dockerfile`: CUDA 12.4 + cuDNN 9 base, Python 3.11 venv managed by **uv**, `buffalo_l` baked into `/models`
- `requirements.txt`: pinned deps (see `.claude/rules/dependencies.md` before touching)
- `compose.yaml`: the main way to run; `run.sh` is the no-compose equivalent (`NO_GPU=1` for CPU)

## Pipeline (sort_faces.py)
1. `list_photos`: recursive `*.jpg|*.jpeg`, case-insensitive
2. `iter_parallel` + `load_image`: `--workers` photos in flight at once (decode + inference; ORT
   sessions are safe for concurrent `run`). JPEG draft-mode decode to `--max-side`, then EXIF transpose.
   On CPU, inference is serialised with a lock (ORT already uses all cores)
3. `detect` / `embed_faces`: SCRFD detection → filter by `--det-thresh` / `--min-face` → one batched
   ArcFace call per photo (not InsightFace's per-face `FaceAnalysis.get`). Bboxes stored in **original** image coords
4. Cache → `faces.npz` (embeddings) + `faces.json` (photos, faces, failures). `--recluster` starts here
5. `cluster` (HDBSCAN, euclidean on unit vectors) → `merge_clusters` (centroid cosine ≥ `--merge-thresh`)
6. `name_clusters`: matches clusters to existing folders via the `.person.json` centroid marker, so
   user-renamed folders keep their names; new people get unused `person_NNN` numbers
7. `write_output`: clears old outputs, copies photos (group photos → every person's folder),
   `_face.jpg` crop per person, `report.csv`, summary

## Commands
```bash
docker compose up --build                                   # full run (needs .env: PHOTOS_DIR, OUTPUT_DIR, UID, GID)
docker compose run --rm face-sorter --recluster --merge-thresh 0.45   # re-group in seconds
NO_GPU=1 ./run.sh <photos> <out> [flags]                    # CPU run without compose
python3 -m py_compile sort_faces.py                         # quick syntax check
```
The dev machine has **no GPU** and no local Python env; test in Docker on CPU with the
`smoke-test` skill. GPU behaviour can only be verified by the user on the server.

## Conventions
- Keep it a single dependency-light script; no new packages without a clear need
- Every CLI flag is documented in README.md (Tuning section) with its default
- Output-affecting changes must keep `--recluster` working against existing `faces.json` caches,
  or say clearly that a full re-run is needed
- User-facing messages go to stdout; warnings/errors prefixed `!! ` to stderr
