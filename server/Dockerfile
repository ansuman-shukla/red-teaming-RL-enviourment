ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env

ENV UV_LINK_MODE=copy \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/data/huggingface/sentence-transformers \
    HF_TOKEN=""

COPY . /app/env

RUN if ! command -v uv >/dev/null 2>&1; then python -m pip install --no-cache-dir uv; fi
RUN uv sync --no-dev --no-editable

FROM ${BASE_IMAGE}

WORKDIR /app/env

ENV PATH="/app/env/.venv/bin:$PATH" \
    PYTHONPATH="/app/env:$PYTHONPATH" \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/data/huggingface/sentence-transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_TOKEN=""

RUN mkdir -p /data/huggingface/transformers /data/huggingface/sentence-transformers

COPY --from=builder /app/env /app/env

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
