FROM python:3.14-slim

# xgboost needs the OpenMP library, which slim images don't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) install dependencies first (cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) then copy the code and the model
COPY . .

ENV PORT=5001
EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health')"

# gunicorn must listen on 0.0.0.0 inside a container
CMD ["gunicorn", "-b", "0.0.0.0:5001", "-w", "2", "--timeout", "60", "app:app"]
