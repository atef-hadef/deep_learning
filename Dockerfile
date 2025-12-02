# Multi-stage build pour réduire la taille de l'image
FROM python:3.11-slim as builder

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage final
FROM python:3.11-slim

WORKDIR /app

# Copier les dépendances Python du builder
COPY --from=builder /root/.local /root/.local

# Copier le code de l'application
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY .env.example .env

# Créer les répertoires nécessaires
RUN mkdir -p data logs models

# S'assurer que les scripts Python sont dans le PATH
ENV PATH=/root/.local/bin:$PATH

# Exposer le port
EXPOSE 8000

# Variable d'environnement pour désactiver le buffering Python
ENV PYTHONUNBUFFERED=1

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
