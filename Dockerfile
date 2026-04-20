FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    fonts-dejavu-core \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Scarica font Roboto Bold (affidabile, leggibile, garantito)
RUN mkdir -p /usr/share/fonts/truetype/roboto && \
    wget -q -O /usr/share/fonts/truetype/roboto/Roboto-Bold.ttf \
    "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf" && \
    fc-cache -f -v

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
