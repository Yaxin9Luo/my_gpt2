# Base image
FROM python:3.11-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY gpt2/ gpt2/
COPY data/ data/
COPY config/ config/


RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
# Change to gpt2 directory before running the script
# Set the working directory to /app/gpt2
WORKDIR /gpt2
CMD ["python3", "train.py", "../config/train_shakespeare_char.py", "--device=cpu", "--compile=False", "--eval_iters=20", "--log_interval=1", "--block_size=64", "--batch_size=12", "--n_layer=4", "--n_head=4", "--n_embd=128", "--max_iters=2000", "--lr_decay_iters=2000", "--dropout=0.0"]
