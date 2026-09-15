FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=/app:/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
COPY training/requirements*.txt /app/training/
RUN pip install --no-cache-dir -r training/requirements.txt
COPY src /app/src
COPY scripts /app/scripts
COPY training /app/training
COPY forecast_store /app/forecast_store
RUN useradd --uid 10002 --create-home trainer && mkdir -p data/raw data/processed results reports/figures logs && chown -R trainer:trainer /app
USER trainer
CMD ["python", "scripts/05_atualizacao_semanal.py", "--pular-lstm"]
