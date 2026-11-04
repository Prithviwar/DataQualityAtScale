# Dockerfile
# -----------
# Pipeline container for the data-quality project.
# Base: Python 3.11-slim; Java 17 installed separately for Spark.

FROM python:3.11-slim

# --- Java 17 (required by PySpark / PyDeequ) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# --- Python deps ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY . .

# Default env
ENV SPARK_VERSION=4.1
ENV PYTHONUNBUFFERED=1

# Entrypoint: override with docker compose run
CMD ["python", "inject_errors.py", "--rows", "100000"]

