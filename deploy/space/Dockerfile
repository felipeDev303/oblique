# Bring your papers on Hugging Face Spaces (Docker SDK). Built from the folder that
# deploy/space/build.py assembles: only the app, its package, the deck and the pages.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY --chown=user . .
RUN uv sync --frozen --no-dev --group product

ENV PATH="/home/user/app/.venv/bin:$PATH"
EXPOSE 7860
# MODEL_API_KEY comes from the Space's secrets, never from a file.
CMD ["python", "-m", "oblique.product.app", "configs/space.yaml"]
