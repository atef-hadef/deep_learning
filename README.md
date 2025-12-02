# 🤖 Analyseur de Sentiments Multi-Plateformes avec IA

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Système intelligent d'analyse de sentiments en temps réel sur les réseaux sociaux (Reddit et Twitter) utilisant des modèles de Deep Learning avancés et génération de résumés avec LLM.

## 🎯 Fonctionnalités Principales

### 🔍 Collecte Intelligente de Données
- **Multi-plateformes** : Reddit et Twitter avec support simultané
- **Filtrage intelligent** : Détection automatique des avis/reviews avec OpinionDetector (CNN)
- **Fetching adaptatif** : Système de pagination intelligent pour obtenir exactement le nombre d'avis souhaités
- **Collecte profonde** : Posts + commentaires avec limite configurable

### 🧠 Analyse IA Avancée
- **Analyse de sentiments** : Classification précise (positif/négatif/neutre) avec RoBERTa fine-tuné
- **Résumés LLM** : Génération de résumés intelligents avec Grok/Claude via OpenRouter API
- **Extraction d'aspects** : Identification automatique des caractéristiques mentionnées
- **Détection d'opinions** : Modèle CNN custom pour filtrer les avis pertinents

### 📊 Visualisation et Tendances
- **Analyse temporelle** : Évolution des mentions et sentiments sur 7-30 jours
- **Graphiques interactifs** : Chart.js avec animations fluides
- **Comparaison multi-plateformes** : Vue unifiée Reddit vs Twitter
- **Dashboard temps réel** : Statistiques agrégées et insights

### 💾 Infrastructure Robuste
- **Base de données** : PostgreSQL avec fallback cache mémoire
- **API REST complète** : Documentation Swagger/OpenAPI automatique
- **Déploiement Docker** : Configuration prête pour production
- **Performance optimisée** : Traitement asynchrone et batch processing

## 📋 Table des Matières

- [🏗️ Architecture](#️-architecture)
- [💻 Technologies](#-technologies)
- [🚀 Installation Rapide](#-installation-rapide)
- [⚙️ Configuration](#️-configuration)
- [📖 Guide d'Utilisation](#-guide-dutilisation)
- [🐳 Déploiement Docker](#-déploiement-docker)
- [📚 Documentation API](#-documentation-api)
- [🎓 Fine-tuning des Modèles](#-fine-tuning-des-modèles)
- [📁 Structure du Projet](#-structure-du-projet)
- [⚠️ Limitations](#️-limitations)
- [🤝 Contribution](#-contribution)
- [📄 License](#-license)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Web (Vue)                       │
│         HTML5 + CSS3 + Vanilla JavaScript + Chart.js        │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Async)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  API Routes  │  │   Services   │  │   Schemas    │      │
│  │   (REST)     │→ │   (Logic)    │→ │  (Pydantic)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────┬────────┬──────────┬──────────┬──────────┬───────────┘
       │        │          │          │          │
       ↓        ↓          ↓          ↓          ↓
┌──────────┐ ┌──────┐ ┌────────┐ ┌──────────┐ ┌────────┐
│  Reddit  │ │Twitter│ │AI/ML   │ │PostgreSQL│ │OpenRouter│
│  (PRAW)  │ │(Tweepy)│ │Models  │ │(asyncpg) │ │  (LLM)   │
└──────────┘ └──────┘ └────────┘ └──────────┘ └────────┘
                         │
            ┌────────────┼────────────┐
            ↓            ↓            ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ RoBERTa  │  │  BART    │  │ CNN      │
    │(Sentiment)│  │(Summary) │  │(Opinion) │
    └──────────┘  └──────────┘  └──────────┘
```

### 🎯 Pipeline de Traitement

1. **Collecte** → Fetching adaptatif multi-plateformes
2. **Filtrage** → OpinionDetector (CNN) + règles heuristiques
3. **Analyse** → RoBERTa sentiments + extraction aspects
4. **Enrichissement** → Résumés BART + insights LLM (optionnel)
5. **Persistance** → PostgreSQL + cache mémoire
6. **Visualisation** → Graphiques temps réel + statistiques

## 💻 Technologies

### Backend Core
- **Framework** : FastAPI 0.104+ (Python 3.9+)
- **API Clients** : 
  - PRAW 7.7+ (Reddit API wrapper)
  - Tweepy 4.16+ (Twitter API v2)
  
### Intelligence Artificielle
- **Transformers** : Hugging Face 🤗
  - `cardiffnlp/twitter-roberta-base-sentiment-latest` - Analyse sentiments
  - `facebook/bart-large-cnn` - Génération résumés
- **TensorFlow/Keras** : Modèle CNN custom pour détection opinions
- **spaCy 3.7+** : NLP (tokenization, POS tagging)
- **OpenRouter API** : Intégration LLM (Grok, Claude, GPT-4)

### Base de Données
- **PostgreSQL 14+** : Persistance principale (asyncpg)
- **Cache mémoire** : Fallback automatique si DB indisponible

### Frontend
- **HTML5/CSS3** : Design responsive avec Flexbox/Grid
- **JavaScript ES6+** : Vanilla JS (pas de framework lourd)
- **Chart.js 4.0+** : Visualisations interactives
- **Font Awesome** : Icônes

### DevOps
- **Docker** : Containerisation
- **Docker Compose** : Orchestration multi-services
- **uvicorn** : Serveur ASGI haute performance

## 🚀 Installation Rapide

### Prérequis

- Python 3.9+ 
- PostgreSQL 14+ (ou Docker)
- Git
- 4GB RAM minimum
- Clés API (voir [Configuration](#️-configuration))

### Installation en 5 Minutes

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/projet_deep_learning.git
cd projet_deep_learning

# 2. Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Télécharger le modèle spaCy
python -m spacy download en_core_web_sm

# 5. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 6. Démarrer l'application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

🎉 **C'est prêt !** Ouvrez http://localhost:8000

### Installation avec Docker (Recommandé)

```bash
# 1. Cloner et configurer
git clone https://github.com/votre-username/projet_deep_learning.git
cd projet_deep_learning
cp .env.example .env
# Éditer .env

# 2. Démarrer tous les services
docker-compose up -d

# 3. Vérifier les logs
docker-compose logs -f app
```

Application disponible sur **http://localhost:8000** 🚀

## ⚙️ Configuration

### 1️⃣ Obtenir les Clés API

#### Reddit API

1. Créer un compte sur [Reddit](https://www.reddit.com)
2. Aller sur https://www.reddit.com/prefs/apps
3. Cliquer sur **"create another app..."**
4. Remplir :
   - **Type** : `script`
   - **Name** : `SocialMediaAnalyzer`
   - **Redirect URI** : `http://localhost:8080`
5. Noter le **client_id** (sous le nom) et **client_secret**

#### Twitter API v2

1. Créer un compte développeur : [Twitter Developer Portal](https://developer.twitter.com)
2. Créer un nouveau projet et une application
3. Générer les clés dans **"Keys and tokens"** :
   - API Key & Secret
   - Bearer Token
   - Access Token & Secret

#### OpenRouter API (Optionnel - pour résumés LLM)

1. Créer un compte sur [OpenRouter](https://openrouter.ai)
2. Aller sur https://openrouter.ai/keys
3. Générer une clé API
4. Ajouter des crédits (à partir de $5)

### 2️⃣ Fichier .env

Créer `.env` à la racine :

```env
# =====================================
# API REDDIT
# =====================================
REDDIT_CLIENT_ID=votre_client_id_reddit
REDDIT_CLIENT_SECRET=votre_secret_reddit
REDDIT_USER_AGENT=SocialMediaAnalyzer/1.0

# =====================================
# API TWITTER v2
# =====================================
TWITTER_BEARER_TOKEN=votre_bearer_token
TWITTER_API_KEY=votre_api_key
TWITTER_API_SECRET=votre_api_secret
TWITTER_ACCESS_TOKEN=votre_access_token
TWITTER_ACCESS_TOKEN_SECRET=votre_token_secret

# =====================================
# DATABASE (PostgreSQL)
# =====================================
# Local
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/social_media_analyzer

# Docker
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/social_media_analyzer

# =====================================
# LLM CONFIGURATION (OpenRouter)
# =====================================
# Optionnel - pour résumés intelligents avec LLM
OPENROUTER_API_KEY=sk-or-v1-votre-cle-ici
OPENROUTER_MODEL=anthropic/claude-3-sonnet

# Modèles disponibles :
# - anthropic/claude-3-sonnet (recommandé, ~$0.003 par résumé)
# - anthropic/claude-3-opus (meilleur qualité, ~$0.015 par résumé)
# - openai/gpt-4-turbo (~$0.01 par résumé)
# - x-ai/grok-beta (si accès disponible)

# =====================================
# APPLICATION SETTINGS
# =====================================
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# =====================================
# AI MODELS
# =====================================
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
SUMMARIZATION_MODEL=facebook/bart-large-cnn

# =====================================
# PERFORMANCE
# =====================================
# Nombre max de posts par requête API
MAX_POSTS_PER_REQUEST=30

# Nombre max de commentaires par post
MAX_COMMENTS_PER_POST=10

# Nombre max de posts pour tendances (par plateforme)
MAX_TRENDS_FETCH_PER_PLATFORM=1500

# Désactiver le zero-shot classifier (plus rapide)
DISABLE_ZERO_SHOT_CLASSIFIER=True
```

### 3️⃣ Configuration PostgreSQL

**Avec Docker** : La base est créée automatiquement ✅

**Installation locale** :

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# macOS (Homebrew)
brew install postgresql@14
brew services start postgresql@14

# Windows
# Télécharger depuis https://www.postgresql.org/download/windows/
```

Créer la base :

```sql
psql -U postgres
CREATE DATABASE social_media_analyzer;
\q
```

Les tables seront créées automatiquement au premier démarrage 🎉

## 📖 Guide d'Utilisation

### Interface Web

#### 1. Recherche d'Avis

1. **Ouvrir** http://localhost:8000
2. **Entrer un mot-clé** : `iPhone 15`, `Tesla Model 3`, `PlayStation 5`...
3. **Sélectionner plateformes** : Reddit, Twitter ou les deux
4. **Choisir période** : 24h, 7 jours, 30 jours
5. **Cliquer "Analyser les avis"**

#### 2. Résultats Détaillés

Vous obtenez :
- 📊 **Distribution sentiments** : Graphique donut interactif
- 📈 **Comparaison plateformes** : Barres Reddit vs Twitter
- 📝 **Liste des posts** : Titre, texte, auteur, date, upvotes
- 💬 **Commentaires** : Sentiments analysés pour chaque commentaire
- 🏷️ **Aspects extraits** : Caractéristiques clés mentionnées

#### 3. Analyse des Tendances

1. **Cliquer "Analyser les tendances"** (après une recherche)
2. **Visualiser** :
   - Évolution temporelle des mentions (graphique ligne)
   - Distribution des sentiments dans le temps
   - Comparaison Reddit vs Twitter

#### 4. Résumé LLM Intelligent

1. **Configurer** `OPENROUTER_API_KEY` dans `.env`
2. **Cliquer "Résumé LLM"** (bouton violet avec 🧠)
3. **Recevoir** un résumé intelligent généré par IA :
   - Satisfaction globale
   - Points négatifs principaux
   - Points positifs principaux
   - Conclusion (positif/mitigé/préoccupant)

### Utilisation via API

#### Recherche de Posts

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "iPhone 15",
    "platforms": ["reddit", "twitter"],
    "limit": 20,
    "time_filter": "week",
    "include_comments": true
  }'
```

#### Analyse Tendances

```bash
curl -X POST "http://localhost:8000/api/trends" \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "iPhone 15",
    "platforms": ["reddit"],
    "time_range": "7d"
  }'
```

#### Résumé LLM

```bash
curl -X GET "http://localhost:8000/api/trends/llm-insight?keyword=iPhone%2015&start_date=2024-12-01&end_date=2024-12-08&platforms=reddit&platforms=twitter"
```

#### Health Check

```bash
curl http://localhost:8000/health
```

### Documentation Interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## 🐳 Déploiement Docker

### Architecture

Le `docker-compose.yml` inclut :
- **app** : Application FastAPI
- **db** : PostgreSQL 14
- **pgadmin** : Interface web PostgreSQL (optionnel)

### Commandes

```bash
# Démarrer
docker-compose up -d

# Logs en temps réel
docker-compose logs -f app

# Arrêter
docker-compose down

# Reconstruire après modifications
docker-compose up -d --build

# Shell du conteneur
docker-compose exec app bash

# Voir les stats
docker stats
```

### URLs

- Application : http://localhost:8000
- API Docs : http://localhost:8000/docs
- PgAdmin : http://localhost:5050 (admin@admin.com / admin)

## 📚 Documentation API

### Endpoints Principaux

#### `POST /api/search`

Recherche et analyse de posts.

**Request Body** :
```json
{
  "keyword": "iPhone 15",
  "platforms": ["reddit", "twitter"],
  "limit": 30,
  "time_filter": "week",
  "include_comments": true
}
```

**Response** (200 OK) :
```json
{
  "keyword": "iPhone 15",
  "platforms": ["reddit", "twitter"],
  "total_posts": 30,
  "posts": [...],
  "overall_sentiment": {
    "positive": 0.65,
    "negative": 0.15,
    "neutral": 0.20,
    "dominant": "positive"
  },
  "execution_time": 8.5
}
```

#### `POST /api/trends`

Analyse des tendances temporelles.

**Request Body** :
```json
{
  "keyword": "iPhone 15",
  "platforms": ["reddit", "twitter"],
  "time_range": "7d"
}
```

#### `GET /api/trends/llm-insight`

Génération résumé intelligent avec LLM.

**Query Parameters** :
- `keyword` : Mot-clé (requis)
- `start_date` : Date ISO format (requis)
- `end_date` : Date ISO format (requis)
- `platforms` : Liste plateformes (optionnel)

#### `GET /health`

Vérification état services.

**Response** (200 OK) :
```json
{
  "status": "healthy",
  "timestamp": "2024-12-02T10:00:00Z",
  "services": {
    "reddit": true,
    "twitter": true,
    "sentiment_model": true,
    "opinion_detector": true,
    "database": true,
    "llm_service": false
  }
}
```

### Codes d'Erreur

- `200` : Succès
- `400` : Requête invalide
- `401` : Non autorisé
- `404` : Ressource non trouvée
- `429` : Rate limit dépassé
- `500` : Erreur serveur interne
- `503` : Service temporairement indisponible

## 🎓 Fine-tuning des Modèles

Le projet inclut un module complet pour améliorer les performances du modèle RoBERTa.

### Quick Start

```bash
# 1. Vérifier l'environnement
python -m app.training.setup_training

# 2. Lancer le fine-tuning (30-45 min GPU / 3-5h CPU)
python -m app.training.train_sentiment_roberta

# 3. Tester le modèle
python -m app.training.test_model
```

### Résultats Attendus

- **Dataset** : tweet_eval/sentiment (45K+ tweets)
- **Amélioration** : +3-5% accuracy
- **Durée** : 30-45 min (GPU) / 3-5h (CPU)
- **Modèle sauvegardé** : `./models/custom-roberta-sentiment/`

### Utiliser le Modèle Fine-tuné

Modifier `.env` :
```env
SENTIMENT_MODEL=./models/custom-roberta-sentiment
```

📖 **Documentation complète** : [app/training/README.md](app/training/README.md)

## 📁 Structure du Projet

```
projet_deep_learning/
│
├── 📁 app/                           # Application principale
│   ├── __init__.py
│   ├── main.py                       # Point d'entrée FastAPI
│   ├── config.py                     # Configuration
│   │
│   ├── 📁 api/                       # Couche API
│   │   ├── __init__.py
│   │   └── routes.py                 # Endpoints REST
│   │
│   ├── 📁 services/                  # Logique métier
│   │   ├── reddit_service.py         # Collecte Reddit
│   │   ├── twitter_service.py        # Collecte Twitter
│   │   ├── sentiment_service.py      # Analyse sentiments
│   │   ├── trends_service.py         # Analyse tendances
│   │   ├── database_service.py       # Persistance PostgreSQL
│   │   ├── opinion_detector.py       # CNN détection opinions
│   │   ├── review_fetcher.py         # Fetching adaptatif
│   │   ├── llm_client.py             # Client OpenRouter
│   │   └── trend_insight_service.py  # Génération insights LLM
│   │
│   ├── 📁 schemas/                   # Schémas Pydantic
│   │   ├── requests.py
│   │   └── responses.py
│   │
│   ├── 📁 training/                  # Module fine-tuning
│   │   ├── train_sentiment_roberta.py
│   │   ├── test_model.py
│   │   └── README.md
│   │
│   └── 📁 models/                    # Modèles ML
│       ├── opinion_detector_model.h5
│       └── opinion_tokenizer.pkl
│
├── 📁 frontend/                      # Interface web
│   ├── index.html
│   ├── 📁 css/
│   │   ├── style.css
│   │   └── trends_styles.css
│   └── 📁 js/
│       ├── app.js
│       └── trends_functions.js
│
├── 📁 tests/                         # Tests unitaires
│   ├── conftest.py
│   └── test_api.py
│
├── .env.example                      # Template variables d'env
├── .gitignore                        # Fichiers ignorés
├── docker-compose.yml                # Orchestration Docker
├── Dockerfile                        # Image Docker app
├── requirements.txt                  # Dépendances production
├── requirements-dev.txt              # Dépendances développement
├── LICENSE                           # Licence MIT
└── README.md                         # Ce fichier
```

## ⚠️ Limitations

### Limites API

#### Twitter API v2 (Free Tier)
- ✋ **500,000 tweets/mois** maximum
- ✋ **50 requêtes/15 min** (endpoint search)
- ✋ **Tweets des 7 derniers jours** uniquement
- 💡 **Solution** : Rate limiting automatique

#### Reddit API
- ✋ **60 requêtes/minute** par IP
- ✋ **1000 posts max** par requête
- 💡 **Solution** : Pagination intelligente

### Ressources Matérielles

- **RAM** : 4-8GB minimum
- **Stockage** : ~2GB (modèles + cache)
- **CPU** : Multi-core recommandé
- **GPU** : Optionnel (accélère fine-tuning 5-10x)
- **Temps traitement** : ~1-2s par post

### Précision Modèles

- **Sentiments** : ~70-75% accuracy
- **Langues** : Anglais principalement
- **Domaines** : Optimisé pour produits tech
- **Sarcasme/ironie** : Détection limitée

### Considérations Éthiques

- ✅ Respect ToS Reddit et Twitter
- ✅ Pas de stockage données personnelles
- ✅ Rate limiting strict
- ✅ Anonymisation auteurs
- ⚠️ Usage académique/recherche recommandé
- ⚠️ Vérifier conditions commerciales avant production

## 🤝 Contribution

Les contributions sont les bienvenues ! 🎉

### Comment contribuer

1. **Fork** le projet
2. **Créer une branche** 
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** vos changements
   ```bash
   git commit -m 'Add: amazing feature'
   ```
4. **Push** vers la branche
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Ouvrir une Pull Request**

### Guidelines

- ✅ Suivre **PEP 8**
- ✅ Ajouter **tests**
- ✅ Documenter avec **docstrings**
- ✅ Mettre à jour **README**
- ✅ Tester localement avant PR

### Idées de Contribution

- 🌍 Support multilingue
- 📊 Nouveaux types de visualisations
- 🤖 Support autres plateformes (YouTube, TikTok...)
- 🧪 Tests end-to-end
- 📱 Application mobile
- 🔒 Authentification utilisateurs

## 📄 License

Distribué sous licence **MIT**. Voir `LICENSE` pour plus d'informations.

Ce projet est développé dans le cadre d'un projet académique à **TEK-UP** - 3ème cycle (2024-2025).

## 👥 Auteurs

**Projet Deep Learning** - TEK-UP 2024-2025

## 🙏 Remerciements

- [Hugging Face 🤗](https://huggingface.co/) - Modèles Transformers
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web
- [Chart.js](https://www.chartjs.org/) - Visualisations
- [OpenRouter](https://openrouter.ai/) - Accès LLMs
- Communautés Reddit et Twitter

## 📞 Support

- 🐛 **Issues** : [GitHub Issues](https://github.com/votre-username/projet_deep_learning/issues)
- 💬 **Discussions** : [GitHub Discussions](https://github.com/votre-username/projet_deep_learning/discussions)
- 📧 **Email** : votre-email@example.com
- 📖 **Documentation** : Ce README + `/app/training/README.md`

---

<p align="center">
  <b>⭐ Si ce projet vous aide, donnez-lui une étoile sur GitHub ! ⭐</b>
</p>

<p align="center">
  Made with ❤️ by TEK-UP Students
</p>
