FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    INSIGHTFACE_ROOT=/models

# Compiler is needed for insightface's Cython extension (no Linux wheels on PyPI).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv --python 3.11

COPY requirements.txt /app/requirements.txt
# insightface's setup.py imports numpy/Cython, so build it without isolation.
RUN uv pip install "numpy>=2,<2.3" cython setuptools \
 && uv pip install --no-build-isolation-package insightface -r /app/requirements.txt

# Bake the buffalo_l model into the image.
RUN mkdir -p /models \
 && python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='/models', providers=['CPUExecutionProvider'])" \
 && chmod -R a+rX /models \
 && rm -rf /tmp/* /tmp/.[!.]*

COPY sort_faces.py /app/sort_faces.py

ENTRYPOINT ["python", "/app/sort_faces.py", "--input", "/photos", "--output", "/out"]
