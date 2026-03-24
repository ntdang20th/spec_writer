# spec-writer/app/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install system deps for tree-sitter (C# code parsing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run interactive Python for learning phases
# Switch to FastAPI in Phase 6: CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
CMD ["python", "-i", "-c", "print('spec-writer ready. Import modules to begin.')"]