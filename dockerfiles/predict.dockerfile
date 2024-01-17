# Base image
FROM python:3.11-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
# Set the working directory to /app
WORKDIR /app
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY gpt2/ gpt2/
COPY data/ data/
COPY config/ config/
COPY out-shakespeare-char/ out-shakespeare-char/
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
# Change to gpt2 directory before running the script
WORKDIR /app/gpt2
# Change to gpt2 directory before running the script

CMD ["python", "inference.py", "--out_dir=/app/out-shakespeare-char", "--device=cpu"]
