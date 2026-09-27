---
paths:
  - "Dockerfile"
  - "requirements.txt"
  - "compose.yaml"
---

# Dependency constraints (learned the hard way; don't "upgrade" past these)

- **onnxruntime-gpu must stay on a CUDA 12 build (≤ 1.26.x).** 1.27+ is built for CUDA 13, which
  dropped Pascal, and the target GPU is a Tesla P4 (sm_61). Check `requires_dist` on PyPI:
  `nvidia-*-cu12` = OK, `cu13` / unsuffixed `nvidia-*~=13` = not OK. Old releases (e.g. 1.20.x) get
  removed from PyPI, so verify a version exists before pinning it.
- The base image must match ORT's CUDA major version and ship cuDNN 9 (`nvidia/cuda:12.x-cudnn-runtime`).
  Driver on the host only needs to be ≥ 525 (CUDA 12 minor-version compatibility).
- **hdbscan ≥ 0.8.44.** Older versions pass `force_all_finite` to scikit-learn, which newer
  scikit-learn versions have removed.
- **insightface has no Linux wheels.** It is built from sdist with
  `--no-build-isolation-package insightface`, so numpy + cython + setuptools must be installed first
  and `build-essential` must stay in the image.
- numpy is `>=2,<2.3`; Python 3.11 comes from uv (`uv venv --python 3.11`), not apt.
- Use **uv** for all installs in Docker (`uv pip install`); don't switch back to pip.
- Keep the `/tmp` cleanup at the end of the model-download layer: a root-owned `/tmp/.config`
  breaks matplotlib for the non-root runtime user.
- The container runs as the host user (`user:` in compose / `--user` in run.sh), so anything the
  runtime needs to write must be under `/tmp` (`HOME=/tmp`) or `/out`.

After changing any of these, rebuild and run the `smoke-test` skill.
