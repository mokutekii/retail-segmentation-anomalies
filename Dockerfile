# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

# The versioned source CSV makes the baseline pipeline self-contained.
COPY online_retail_II.csv ./online_retail_II.csv
RUN mkdir -p /app/artifacts

ENTRYPOINT ["python", "-m", "retail_segmentation.train"]
CMD ["--fast"]
