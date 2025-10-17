FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances si présents
COPY requirements.txt pyproject.toml* poetry.lock* ./ 2>/dev/null || true

# Installe les deps si requirements.txt existe, sinon installe un minimum viable
RUN if [ -f "requirements.txt" ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir fastapi uvicorn[standard] scikit-learn pandas numpy mlflow; \
    fi

# Copie du reste du projet (API + code)
COPY . /app

# On s'assure que le modèle exporté est présent dans l'image
# (tu l'as créé via export_model.py => dossier local "model/")
# On fixe l'URI pour que app_loan.py charge localement
ENV MLFLOW_MODEL_URI=/app/model

EXPOSE 5000
CMD ["uvicorn", "app_loan:app", "--host", "0.0.0.0", "--port", "5000"]
