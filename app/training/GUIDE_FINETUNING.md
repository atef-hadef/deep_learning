# Guide Complet - Fine-tuning RoBERTa

Ce guide détaillé vous accompagne pas à pas dans le processus de fine-tuning du modèle RoBERTa pour améliorer les performances d'analyse de sentiments.

---

## 📋 Table des Matières

1. [Pourquoi Fine-tuner ?](#pourquoi-fine-tuner)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Étape 1 : Préparer l'environnement](#étape-1--préparer-lenvironnement)
5. [Étape 2 : Lancer le fine-tuning](#étape-2--lancer-le-fine-tuning)
6. [Étape 3 : Surveiller l'entraînement](#étape-3--surveiller-lentraînement)
7. [Étape 4 : Tester le modèle](#étape-4--tester-le-modèle)
8. [Étape 5 : Intégrer le modèle](#étape-5--intégrer-le-modèle)
9. [Optimisation et Tuning](#optimisation-et-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Pourquoi Fine-tuner ?

### Avantages du Fine-tuning

✅ **Performance améliorée** : Adaptation au vocabulaire spécifique de votre domaine  
✅ **Meilleure précision** : Réduction des erreurs sur vos cas d'usage  
✅ **Cohérence** : Comportement plus prévisible et stable  
✅ **Personnalisation** : Adaptation aux nuances de vos données  

### Quand Fine-tuner ?

- Votre domaine a un vocabulaire spécifique (produits tech, gaming, etc.)
- Le modèle de base fait des erreurs récurrentes
- Vous avez accès à des données annotées de qualité
- Vous voulez maximiser les performances

---

## Prérequis

### Matériel Recommandé

| Configuration | CPU | GPU | Durée Estimée |
|--------------|-----|-----|---------------|
| **Minimale** | 4 cores | - | ~4-5 heures |
| **Recommandée** | 8 cores | GTX 1060 (6GB) | ~45 min |
| **Optimale** | 16+ cores | RTX 3060+ (12GB) | ~20-30 min |

### Espace Disque

- Dataset : ~500 MB
- Modèle base : ~500 MB
- Modèle fine-tuné : ~500 MB
- Checkpoints temporaires : ~1-2 GB
- **Total recommandé : 5 GB libres**

### Software

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (si GPU disponible)
- transformers 4.35+

---

## Installation

### 1. Installer les dépendances

```bash
cd projet_deep_learning

# Installer les packages de base (déjà fait normalement)
pip install -r requirements.txt

# Vérifier que datasets est installé
pip install datasets accelerate
```

### 2. Vérifier CUDA (GPU)

```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Sortie attendue :**
```
CUDA disponible: True
Device: NVIDIA GeForce RTX 3060
```

Si `False`, l'entraînement utilisera le CPU (plus lent mais fonctionnel).

### 3. Télécharger spaCy model (si pas déjà fait)

```bash
python -m spacy download en_core_web_sm
```

---

## Étape 1 : Préparer l'environnement

### 1.1 Vérifier la structure

```bash
# Votre arborescence devrait ressembler à :
projet_deep_learning/
├── training/
│   ├── __init__.py
│   ├── train_sentiment_roberta.py
│   ├── test_model.py
│   └── README.md
├── models/
│   └── (vide pour l'instant)
├── app/
├── requirements.txt
└── ...
```

### 1.2 Créer le dossier models (si nécessaire)

```bash
mkdir -p models
```

### 1.3 Test de connexion Hugging Face

```bash
python -c "from datasets import load_dataset; print('✅ Connection OK')"
```

---

## Étape 2 : Lancer le fine-tuning

### Mode 1 : Entraînement Complet (Production)

```bash
cd projet_deep_learning
python -m training.train_sentiment_roberta
```

**Ce qui va se passer :**

1. ⏬ Téléchargement du dataset tweet_eval (~500 MB) - **1ère fois uniquement**
2. ⏬ Chargement du modèle RoBERTa de base (~500 MB) - **1ère fois uniquement**
3. 🔄 Tokenisation des 45,615 tweets d'entraînement (~2-3 min)
4. 🚀 **Entraînement sur 3 époques** :
   - Epoch 1/3 : ~15-20 min (GPU) ou ~1h30 (CPU)
   - Epoch 2/3 : ~15-20 min (GPU) ou ~1h30 (CPU)
   - Epoch 3/3 : ~15-20 min (GPU) ou ~1h30 (CPU)
5. 💾 Sauvegarde du meilleur modèle
6. 📊 Évaluation sur test set (12,284 tweets)

**Sortie attendue (exemple) :**

```
================================================================================
🚀 FINE-TUNING ROBERTA POUR ANALYSE DE SENTIMENTS
================================================================================

🔹 Chargement du dataset tweet_eval/sentiment ...
   - Train samples: 45615
   - Validation samples: 2000
   - Test samples: 12284

🔹 Chargement du modèle de base : cardiffnlp/twitter-roberta-base-sentiment-latest
✅ Modèle et tokenizer chargés

🔹 Tokenisation du dataset ...
✅ Tokenisation terminée

🔹 Configuration de l'entraînement ...
   - Output directory: ./models/custom-roberta-sentiment
   - Epochs: 3
   - Batch size: 16
   - Learning rate: 2e-05

================================================================================
🚀 LANCEMENT DU FINE-TUNING
================================================================================

Epoch 1/3:
[====================] 2851/2851 [15:23<00:00, 3.08it/s]
{'loss': 0.4123, 'learning_rate': 1.5e-05, 'epoch': 1.0}
{'eval_loss': 0.3892, 'eval_accuracy': 0.7150, 'eval_f1_macro': 0.6923}

Epoch 2/3:
[====================] 2851/2851 [15:21<00:00, 3.09it/s]
{'loss': 0.3456, 'learning_rate': 1e-05, 'epoch': 2.0}
{'eval_loss': 0.3721, 'eval_accuracy': 0.7285, 'eval_f1_macro': 0.7045}

Epoch 3/3:
[====================] 2851/2851 [15:19<00:00, 3.10it/s]
{'loss': 0.3102, 'learning_rate': 5e-06, 'epoch': 3.0}
{'eval_loss': 0.3698, 'eval_accuracy': 0.7310, 'eval_f1_macro': 0.7089}

✅ Entraînement terminé. Sauvegarde du modèle ...
✅ Modèle sauvegardé dans ./models/custom-roberta-sentiment

================================================================================
📊 ÉVALUATION SUR LE JEU DE TEST
================================================================================

📊 Métriques finales (test set):
------------------------------------------------------------
  eval_loss.................................... 0.3645
  eval_accuracy................................ 0.7321
  eval_f1_macro................................ 0.7102
  eval_precision_macro......................... 0.7145
  eval_recall_macro............................ 0.7089
------------------------------------------------------------

✅ FINE-TUNING TERMINÉ AVEC SUCCÈS

📁 Modèle disponible dans: ./models/custom-roberta-sentiment
```

### Mode 2 : Test Rapide (Développement)

Pour un test rapide (~15 min sur CPU), modifiez `training/train_sentiment_roberta.py` :

```python
# Ligne 23
USE_SUBSET = True  # ← Changer False à True
```

Puis lancez :
```bash
python -m training.train_sentiment_roberta
```

**Utilise seulement :**
- 10,000 tweets pour train (au lieu de 45,615)
- 2,000 tweets pour validation (au lieu de 2,000)

---

## Étape 3 : Surveiller l'entraînement

### Option 1 : Logs en temps réel

Les logs s'affichent automatiquement dans le terminal.

### Option 2 : TensorBoard (Recommandé)

Dans un **nouveau terminal** :

```bash
cd projet_deep_learning
tensorboard --logdir models/custom-roberta-sentiment/logs
```

Ouvrir dans navigateur : **http://localhost:6006**

**Graphiques disponibles :**
- 📉 Loss (train & validation) par époque
- 📈 Accuracy, F1-macro évolution
- ⏱️ Learning rate schedule
- 🔢 Gradient norms

### Option 3 : Fichier training_info.txt

Après entraînement, consulter :
```bash
cat models/custom-roberta-sentiment/training_info.txt
```

---

## Étape 4 : Tester le modèle

### Test Automatique (Comparaison)

```bash
python -m training.test_model
```

**Sortie exemple :**

```
📊 COMPARAISON DES PRÉDICTIONS
════════════════════════════════════════════════════════════════════════

Test #1: I absolutely love this product! Best purchase ever! 😍
────────────────────────────────────────────────────────────────────────

📌 Modèle BASE:
   Sentiment: POSITIVE (confidence: 0.892)
   Scores: Pos=0.892 | Neu=0.085 | Neg=0.023

🎯 Modèle FINE-TUNED:
   Sentiment: POSITIVE (confidence: 0.947)
   Scores: Pos=0.947 | Neu=0.042 | Neg=0.011

📈 Amélioration de confiance: 0.892 → 0.947 (+0.055)
```

### Test sur Texte Personnalisé

```bash
python -m training.test_model "The camera is great but battery sucks"
```

### Test Interactif Python

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle fine-tuné
tokenizer = AutoTokenizer.from_pretrained("./models/custom-roberta-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("./models/custom-roberta-sentiment")

# Test
text = "Amazing product! Highly recommended!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(f"Negative: {probs[0][0]:.3f}")
print(f"Neutral:  {probs[0][1]:.3f}")
print(f"Positive: {probs[0][2]:.3f}")
```

---

## Étape 5 : Intégrer le modèle

### Méthode 1 : Modifier config.py (Recommandé)

Éditer `app/config.py` :

```python
class Settings(BaseSettings):
    # ... autres paramètres ...
    
    # AVANT:
    # sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # APRÈS:
    sentiment_model: str = "./models/custom-roberta-sentiment"
```

### Méthode 2 : Variable d'environnement

Éditer `.env` :

```bash
# Modèle fine-tuné (local)
SENTIMENT_MODEL=./models/custom-roberta-sentiment

# Ou modèle de base (Hugging Face)
# SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
```

### Redémarrer l'application

```bash
uvicorn app.main:app --reload
```

**Vérifier dans les logs :**
```
INFO: Chargement du modèle de sentiment depuis ./models/custom-roberta-sentiment
INFO: ✅ Modèle chargé avec succès
```

### Tester via API

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "iPhone 15",
    "platforms": ["reddit"],
    "limit": 5
  }'
```

---

## Optimisation et Tuning

### Améliorer la Précision

#### 1. Augmenter le nombre d'époques

```python
NUM_EPOCHS = 5  # Au lieu de 3
```

⚠️ Risque d'overfitting si > 5

#### 2. Ajuster le learning rate

```python
# Plus petit = entraînement plus stable mais plus lent
LR = 1e-5  # Au lieu de 2e-5

# Plus grand = plus rapide mais risque d'instabilité
LR = 3e-5
```

#### 3. Augmenter la longueur de séquence

```python
MAX_SEQ_LENGTH = 256  # Au lieu de 128
```

⚠️ Consomme 2x plus de mémoire

#### 4. Ajouter vos propres données

Créer `custom_data.json` :

```json
[
  {"text": "Love the new iPhone camera!", "label": 2},
  {"text": "Battery drains too fast", "label": 0},
  {"text": "It's okay, nothing special", "label": 1}
]
```

Modifier `train_sentiment_roberta.py` pour charger ces données supplémentaires.

### Accélérer l'Entraînement

#### 1. Mixed Precision (FP16)

```python
training_args = TrainingArguments(
    ...
    fp16=True,  # ← Activer (nécessite GPU Volta+ ou Ampere)
)
```

Gain : **~40% plus rapide**, même consommation mémoire

#### 2. Gradient Accumulation

```python
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=8,  # Réduire
    gradient_accumulation_steps=2,  # ← Ajouter
)
```

Simule `batch_size=16` avec moins de mémoire.

#### 3. Augmenter Batch Size (si mémoire suffisante)

```python
BATCH_SIZE = 32  # Au lieu de 16
```

Gain : **~20% plus rapide**

---

## Troubleshooting

### ❌ Erreur : CUDA out of memory

**Symptôme :**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions :**

1. Réduire batch size
```python
BATCH_SIZE = 8  # Ou même 4
```

2. Réduire max_seq_length
```python
MAX_SEQ_LENGTH = 64
```

3. Gradient checkpointing
```python
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
)
```

4. Utiliser CPU
```python
training_args = TrainingArguments(
    ...
    no_cuda=True,
)
```

---

### ❌ Erreur : ConnectionError downloading dataset

**Symptôme :**
```
ConnectionError: Couldn't reach https://huggingface.co/datasets/...
```

**Solutions :**

1. Vérifier connexion internet
```bash
ping huggingface.co
```

2. Utiliser proxy (si nécessaire)
```bash
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080
```

3. Télécharger manuellement
```python
from datasets import load_dataset
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", cache_dir="./cache")
```

---

### ❌ Accuracy ne s'améliore pas

**Symptôme :**
Après 3 époques, accuracy stagne à ~0.50-0.60

**Diagnostics :**

1. **Dataset déséquilibré** : Vérifier distribution des labels
```python
from collections import Counter
labels = [ex['label'] for ex in dataset['train']]
print(Counter(labels))
```

2. **Learning rate trop élevé** : Réduire à `1e-5`

3. **Underfitting** : Augmenter époques à 5

4. **Overfitting** : Vérifier `eval_loss` :
   - Si `train_loss` ↓ mais `eval_loss` ↑ → overfitting

---

### ⚠️ Warning : Some weights not initialized

**Symptôme :**
```
Some weights of the model checkpoint at ... were not used when initializing...
```

**Explication :** Normal pour fine-tuning. Le modèle adapte ses poids.

**Action :** Aucune (c'est attendu)

---

## Métriques de Référence

### Modèle de Base (sans fine-tuning)

| Métrique | Score |
|----------|-------|
| Accuracy | ~0.695 |
| F1 Macro | ~0.675 |
| Precision | ~0.680 |
| Recall | ~0.670 |

### Modèle Fine-tuné (attendu)

| Métrique | Score | Amélioration |
|----------|-------|--------------|
| Accuracy | ~0.720-0.740 | +3-5% |
| F1 Macro | ~0.700-0.720 | +3-5% |
| Precision | ~0.710-0.730 | +3-5% |
| Recall | ~0.700-0.720 | +3-5% |

**Objectif réaliste :** +3-5% d'amélioration sur toutes les métriques

---

## Ressources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Tweet Eval Dataset](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

---

**Dernière mise à jour :** 16 novembre 2025
