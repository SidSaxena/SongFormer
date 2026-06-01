FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System deps (librosa needs libsndfile, ffmpeg for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Python deps (torch comes from base image)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source
COPY . .

# Pre-download checkpoints (optional — comment out to download at first run)
# RUN python -c "import os; os.chdir('src/SongFormer'); from utils.fetch_pretrained import download_all; download_all()"

WORKDIR /app/src/SongFormer

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "/app/app.py"]
