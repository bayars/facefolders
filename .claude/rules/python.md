---
paths:
  - "**/*.py"
---

# Python style

- Target Python 3.11 (the container's uv-managed interpreter).
- Heavy imports (`insightface`, `onnxruntime`, `hdbscan`) stay inside the functions that use them,
  so `--help` and `--recluster` start fast and don't touch the GPU unnecessarily.
- Images are BGR `uint8` numpy arrays (InsightFace convention); convert only at PIL boundaries.
- Face bboxes are stored in original (EXIF-transposed, full-resolution) coordinates; divide by
  the `scale` from `load_image` when producing them.
- Embeddings are L2-normalised float32; similarity = dot product. Re-normalise after averaging
  (`centroid()`).
- Don't let one bad photo abort a run: record it in `failed` and continue.
- Match the existing style: short docstrings only where the *why* isn't obvious, no type-hint
  retrofits, f-strings, `pathlib`.
