# ==========================================
# Stage 1: Build & Dependencies Preparation
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies in a local user directory (reduces package footprint)
RUN pip install --no-cache-dir --user -r requirements.txt


# ==========================================
# Stage 2: Final Secure Runner Image
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /backend

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /backend
ENV PATH=/home/appuser/.local/bin:$PATH

# Install only runtime essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Add a dedicated non-root application user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -m -d /home/appuser appuser

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

# Set appropriate directory ownership permissions
RUN chown -R appuser:appgroup /backend /home/appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
