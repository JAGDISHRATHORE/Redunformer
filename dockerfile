FROM python:3.12-slim

# Keep Python output unbuffered and avoid pip cache
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools and dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Install the package and runtime dependencies
RUN pip install .

# Create output directory
RUN mkdir -p /app/experiments

ENTRYPOINT ["python", "scripts/run_baseline.py"]