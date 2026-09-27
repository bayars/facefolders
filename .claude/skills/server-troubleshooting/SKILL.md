---
name: server-troubleshooting
description: Diagnose problems the user reports from running facefolders on their GPU server (Tesla P4 + Ollama, Docker Compose), such as permission errors, "Running on CPU", CUDA or cuDNN errors, out-of-memory errors, slow runs or build failures. Use when the user pastes container logs or an error from the server.
---

# Troubleshooting on the server

You can't reach the server. Work from the logs the user pastes, and give them commands to run themselves. Keep the commands read-only unless a fix is clearly needed.

| Log / symptom | Cause | Fix |
|---|---|---|
| `Cannot write to output dir … uid:gid` / `PermissionError: /out/...` | OUTPUT_DIR is owned by root (Docker created it) or UID/GID don't match | `sudo chown -R $(id -u):$(id -g) OUTPUT_DIR`; put `UID=`/`GID=` from `id -u`/`id -g` in `.env` |
| `!! No NVIDIA device visible` | Container started without the GPU | Use compose (it has the `deploy.resources` GPU reservation) or `--gpus all`; check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |
| `could not select device driver "nvidia"` | `nvidia-container-toolkit` not installed or configured | Install it, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `!! GPU init failed … falling back to CPU`, `CUDA_ERROR_NO_BINARY_FOR_GPU`, `no kernel image` | onnxruntime build without sm_61 kernels (CUDA 13 build) | See `.claude/rules/dependencies.md`; pin a CUDA 12 onnxruntime-gpu |
| `CUDA out of memory` / `Failed to allocate` | Ollama is holding the P4's 8 GB | `nvidia-smi` to see usage; `ollama ps` then `ollama stop <model>` and re-run |
| `Using GPU` but below ~5 photos/s | Photo decoding on CPU, or slow disk / network share | Try `--workers 8`; check whether PHOTOS_DIR is on NFS or SMB |
| `Copying` step slow | Slow target disk; each photo is copied once per person | Expected on HDD/NAS; the copy is plain `shutil.copy2` |
| Exit code 139 / segfault at start | CUDA provider started without a device | Should be caught by the `/dev/nvidiactl` check; if not, send the full log |
| `No JPEG files found` | Wrong PHOTOS_DIR or `.env` not picked up | `docker compose config` shows the resolved mounts |

Helpful commands to give the user:
```bash
docker compose config            # resolved paths, user and GPU reservation
nvidia-smi; ollama ps            # GPU memory and loaded models
id -u; id -g; ls -ld "$OUTPUT_DIR"
```
If you change code for a fix, run the `smoke-test` skill before handing it back.
