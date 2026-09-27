---
name: smoke-test
description: Build the Docker image and run facefolders end-to-end on CPU against synthetic photos, then assert the results (6 people found, EXIF rotation, no-face and broken files, subfolders, renamed folders keep names). Use after any change to sort_faces.py, Dockerfile or requirements.txt, and before committing.
---

# Smoke test

The dev machine has no GPU, so this runs on CPU inside the real image. It takes about 1 minute after the image is built.

Run from the repo root. Replace `$S` with the session scratchpad directory. Don't use real photos.

```bash
S=<scratchpad>/smoke; rm -rf $S; mkdir -p $S/photos $S/out
SK=.claude/skills/smoke-test
U="$(id -u):$(id -g)"

docker build -q -t face-sorter .

# 1. Fixtures (generated inside the image, which has insightface's sample data)
docker run --rm --user $U --entrypoint python -v $S/photos:/p -v $PWD/$SK:/sk:ro face-sorter /sk/make_photos.py /p

# 2. Full run on CPU (no --gpus, so the script must detect the missing device and fall back)
docker run --rm --user $U -v $S/photos:/photos:ro -v $S/out:/out face-sorter

# 3. Assertions
docker run --rm --user $U --entrypoint python -v $S/out:/out -v $PWD/$SK:/sk:ro face-sorter /sk/check_output.py /out

# 4. Name persistence: rename a folder, recluster, the name must survive
mv $S/out/person_002 $S/out/Alice
docker run --rm --user $U -v $S/photos:/photos:ro -v $S/out:/out face-sorter --recluster
test -d $S/out/Alice && ! test -d $S/out/person_002 && echo "OK: rename kept"
docker run --rm --user $U --entrypoint python -v $S/out:/out -v $PWD/$SK:/sk:ro face-sorter /sk/check_output.py /out
```

Expected output:
- Step 2 prints `!! No NVIDIA device visible` and `!! Running on CPU`. That is correct here.
- The summary shows `People (clusters): 6`. There may be `merging clusters` lines when HDBSCAN splits the mirrored copies. Either way is fine as long as the final count is 6.
- Every `check_output.py` run prints `OK`.

If a step fails, look at the output of step 2 before changing the checks. The checks describe the intended behaviour.

GPU paths (CUDA init, the VRAM fallback) can't be tested here. Say so in your report, and ask the user to run `docker compose up --build` on the server and look for `Using GPU`.
