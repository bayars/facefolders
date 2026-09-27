# Face sorter

Sorts JPEG photos into per-person folders using InsightFace (`buffalo_l`) + HDBSCAN.
Runs in Docker with CUDA 12 (works on Pascal GPUs like the Tesla P4).

## Prerequisites (on the GPU server)
- NVIDIA driver (already there if Ollama uses the GPU) and `nvidia-container-toolkit`
- Check: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
- ~1.5 GB free VRAM. If Ollama has a big model loaded: `ollama stop <model>`

## Usage
```bash
./run.sh /path/to/photos /path/to/output
```
Or with Docker Compose (put the variables in a `.env` file to avoid retyping them):
```bash
mkdir -p /path/to/output   # create it first, otherwise Docker creates it owned by root
PHOTOS_DIR=/path/to/photos OUTPUT_DIR=/path/to/output docker compose up --build
# extra flags:
PHOTOS_DIR=... OUTPUT_DIR=... docker compose run --rm face-sorter --recluster --min-samples 1
```
Files are written as UID/GID 1000 by default. If your user has a different ID, set `UID` and `GID` in `.env`.

Output:
- `person_001/ …`: one folder per person, largest first. Group photos are copied into every matching person's folder. `_face.jpg` in each folder shows who it is.
- `unknown/`: photos whose faces couldn't be grouped (e.g. people seen only once or twice)
- `no_faces/`: photos without faces
- `report.csv`: per-photo face count and persons
- `faces.npz` / `faces.json`: cached embeddings

Photos from subfolders are named `sub__dir__file.jpg`. Originals are never modified.

## Naming people
Rename a folder, e.g. `person_003` → `Ali`. On the next run (including `--recluster`) the
same person is put back into `Ali/`. This works because each person folder has a hidden
`.person.json` holding that person's average face, which moves with the folder.
Unnamed people also keep their `person_NNN` number between runs.

## Tuning
Re-cluster without re-running detection (takes seconds):
```bash
./run.sh /path/to/photos /path/to/output --recluster --min-cluster-size 4 --min-samples 1
```
- One person split into several folders → lower `--merge-thresh` (default 0.5, e.g. 0.4), or lower `--min-samples`
- Different people merged (e.g. siblings) → raise `--merge-thresh` (e.g. 0.6; `1` disables merging), or raise `--min-samples` / `--min-cluster-size`
- Too many junk faces (tiny, blurry, background) → raise `--det-thresh` (0.6) or `--min-face` (40 px)

Speed: `--workers` (default 4) is how many photos are processed at once. On the GPU, raise it
(e.g. 6–8) if `nvidia-smi` shows `GPU-Util` well below 100% during detection. GPU memory
use (~0.5–1 GB) is normal and doesn't limit speed.

Each run replaces all person folders (renamed ones too), `unknown` and `no_faces` in the
output directory. Don't keep your own files in them.
Set `NO_GPU=1` to run on a machine without a GPU.
