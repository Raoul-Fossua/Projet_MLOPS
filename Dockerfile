FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Paquets système minimaux
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python — si requirements.txt existe
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "requirements.txt absent → install minimal" && \
     pip install --no-cache-dir fastapi uvicorn[standard] scikit-learn pandas numpy mlflow)

# Code + modèle exporté (dossier model/)
COPY . /app

# L’API charge le modèle packagé dans /app/model
ENV MLFLOW_MODEL_URI=/app/model

EXPOSE 5000
CMD ["uvicorn", "app_loan:app", "--host", "0.0.0.0", "--port", "5000"]
