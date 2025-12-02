# Training Module - Fine-tuning RoBERTa

Ce module contient les scripts pour fine-tuner le modèle RoBERTa sur des données d'analyse de sentiments.

## Structure

```
training/
├── __init__.py
└── train_sentiment_roberta.py    # Script principal de fine-tuning
```

## Prérequis

```bash
pip install datasets scikit-learn
```

Tous les autres packages (transformers, torch, etc.) sont déjà dans `requirements.txt`.

## Lancement du Fine-tuning

### Option 1 : Entraînement complet (recommandé)

```bash
cd projet_deep_learning
python -m training.train_sentiment_roberta
```

**Durée estimée :**
- GPU (CUDA) : ~30-45 minutes
- CPU : ~3-5 heures

### Option 2 : Test rapide (subset)

Modifier dans `train_sentiment_roberta.py` :
```python
USE_SUBSET = True  # Ligne 23
```

Puis lancer :
```bash
python -m training.train_sentiment_roberta
```

**Durée estimée :** ~10-15 minutes (CPU)

## Configuration

### Hyperparamètres modifiables

Dans `train_sentiment_roberta.py` :

```python
BASE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "./models/custom-roberta-sentiment"
MAX_SEQ_LENGTH = 128        # Longueur max tokens (augmenter si textes longs)
NUM_EPOCHS = 3              # Nombre d'époques (3-5 recommandé)
BATCH_SIZE = 16             # Taille batch (réduire si mémoire insuffisante)
LR = 2e-5                   # Learning rate
```

### GPU/CPU

Le script détecte automatiquement le GPU. Pour forcer CPU :
```python
training_args = TrainingArguments(
    ...
    no_cuda=True,  # Ajouter cette ligne
)
```

## Dataset Utilisé

**tweet_eval/sentiment** (Cardiff NLP)
- **Train**: 45,615 tweets
- **Validation**: 2,000 tweets
- **Test**: 12,284 tweets

**Labels:**
- 0 = Negative
- 1 = Neutral
- 2 = Positive

## Résultats Attendus

Après fine-tuning, vous devriez voir :

```
models/custom-roberta-sentiment/
├── config.json                 # Configuration modèle
├── pytorch_model.bin           # Poids du modèle (500MB)
├── tokenizer.json              # Tokenizer
├── vocab.json                  # Vocabulaire
├── merges.txt                  # BPE merges
├── special_tokens_map.json     # Tokens spéciaux
├── training_info.txt           # Info entraînement
└── logs/                       # TensorBoard logs
    └── events.out.tfevents.*
```

### Métriques typiques (test set)

```
Accuracy:        ~0.70-0.72
F1 Macro:        ~0.68-0.70
Precision Macro: ~0.68-0.71
Recall Macro:    ~0.68-0.70
```

## Visualiser les Logs (TensorBoard)

```bash
tensorboard --logdir models/custom-roberta-sentiment/logs
```

Ouvrir : http://localhost:6006

## Utiliser le Modèle Fine-tuné

### Méthode 1 : Modifier la configuration

Dans `app/config.py` :
```python
class Settings(BaseSettings):
    sentiment_model: str = "./models/custom-roberta-sentiment"  # Chemin local
    # ou
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Original
```

### Méthode 2 : Variable d'environnement

Dans `.env` :
```bash
SENTIMENT_MODEL=./models/custom-roberta-sentiment
```

Redémarrer l'application :
```bash
uvicorn app.main:app --reload
```

## Troubleshooting

### Erreur : CUDA out of memory

**Solution 1** : Réduire batch size
```python
BATCH_SIZE = 8  # Au lieu de 16
```

**Solution 2** : Utiliser gradient accumulation
```python
training_args = TrainingArguments(
    ...
    gradient_accumulation_steps=2,  # Simule batch_size * 2
)
```

### Erreur : Dataset download fails

```bash
# Télécharger manuellement
export HF_DATASETS_OFFLINE=0
python -c "from datasets import load_dataset; load_dataset('cardiffnlp/tweet_eval', 'sentiment')"
```

### Performance CPU très lente

Activer optimisations PyTorch :
```python
import torch
torch.set_num_threads(4)  # Nombre de CPU cores
```

## Améliorations Futures

- [ ] Ajout de données custom (Reddit/Twitter spécifiques)
- [ ] Data augmentation (backtranslation, paraphrasing)
- [ ] Hyperparameter tuning (Ray Tune, Optuna)
- [ ] Distillation vers modèle plus petit (DistilRoBERTa)
- [ ] Multi-task learning (sentiment + aspects)
- [ ] Continual learning sur nouvelles données
