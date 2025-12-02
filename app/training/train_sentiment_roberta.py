# training/train_sentiment_roberta.py
"""
Script de fine-tuning du modèle RoBERTa pour l'analyse de sentiments.
Utilise le dataset tweet_eval (cardiffnlp) pour affiner le modèle de base.

Labels:
  0 -> negative
  1 -> neutral
  2 -> positive
"""

import os
import numpy as np
from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ===========================
# Configuration de base
# ===========================
BASE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "./models/custom-roberta-sentiment"  # dossier final du modèle

# 🔧 Adapté pour petit GPU (MX350)
MAX_SEQ_LENGTH = 96          # Optimal pour tweets (plus rapide, moins de VRAM)
NUM_EPOCHS = 1               # Commencer avec 1 epoch pour test GPU
PER_DEVICE_BATCH_SIZE = 2    # Très petit batch par GPU (limite VRAM)
GRAD_ACC_STEPS = 8           # 2 * 8 = batch effectif de 16
LR = 2e-5

# Optionnel : réduire taille dataset pour tests rapides
USE_SUBSET = False           # False = dataset complet (45K tweets) - RECOMMANDÉ pour GPU
TRAIN_SUBSET_SIZE = 10000    # Utilisé seulement si USE_SUBSET=True
EVAL_SUBSET_SIZE = 2000      # Utilisé seulement si USE_SUBSET=True


def load_tweet_eval_sentiment():
    """
    Charge le dataset 'tweet_eval' (task 'sentiment').

    Returns:
        DatasetDict contenant train, validation, test

    Labels:
      0 -> negative
      1 -> neutral
      2 -> positive
    """
    print("🔹 Chargement du dataset tweet_eval/sentiment ...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Validation samples: {len(dataset['validation'])}")
    print(f"   - Test samples: {len(dataset['test'])}")

    return dataset


def preprocess_dataset(dataset, tokenizer):
    """
    Tokenisation du texte pour RoBERTa.

    Args:
        dataset: Dataset Hugging Face
        tokenizer: Tokenizer RoBERTa

    Returns:
        Dataset tokenisé et formaté pour PyTorch
    """
    print("🔹 Tokenisation du dataset ...")

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)

    # Hugging Face Trainer attend ces colonnes :
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    print("✅ Tokenisation terminée")
    return tokenized


def compute_metrics(eval_pred):
    """
    Métriques pour validation/test : accuracy + F1 macro.

    Args:
        eval_pred: Tuple (logits, labels)

    Returns:
        Dict avec accuracy, f1_macro, precision_macro, recall_macro
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def save_training_info(output_dir, metrics):
    """
    Sauvegarde informations sur l'entraînement dans un fichier texte.

    Args:
        output_dir: Répertoire de sortie
        metrics: Dictionnaire des métriques finales
    """
    info_file = os.path.join(output_dir, "training_info.txt")

    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("INFORMATIONS SUR L'ENTRAÎNEMENT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modèle de base: {BASE_MODEL_NAME}\n")
        f.write(f"Dataset: tweet_eval/sentiment\n\n")

        f.write("Hyperparamètres:\n")
        f.write(f"  - Max sequence length: {MAX_SEQ_LENGTH}\n")
        f.write(f"  - Epochs: {NUM_EPOCHS}\n")
        f.write(f"  - Per-device batch size: {PER_DEVICE_BATCH_SIZE}\n")
        f.write(f"  - Gradient accumulation steps: {GRAD_ACC_STEPS}\n")
        f.write(f"  - Learning rate: {LR}\n")
        f.write(f"  - Weight decay: 0.01\n\n")

        if metrics:
            f.write("Métriques finales (test set):\n")
            for key, value in metrics.items():
                f.write(f"  - {key}: {value:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"📄 Informations d'entraînement sauvegardées dans {info_file}")


def count_trainable_parameters(model):
    """
    Retourne le nombre de paramètres entraînables (pour info logs).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Fonction principale d'entraînement.
    """
    print("=" * 80)
    print("🚀 FINE-TUNING ROBERTA POUR ANALYSE DE SENTIMENTS")
    print("=" * 80 + "\n")

    # 1) Charger dataset
    raw_dataset = load_tweet_eval_sentiment()

    # 2) Charger tokenizer + modèle RoBERTa existant
    print(f"\n🔹 Chargement du modèle de base : {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,  # négatif / neutre / positif
    )
    print("✅ Modèle et tokenizer chargés")

    # 🔧 IMPORTANT : geler le backbone RoBERTa pour économiser la VRAM
    print("\n🧊 Gel du backbone RoBERTa (on entraîne seulement la tête de classification)...")
    for param in model.roberta.parameters():
        param.requires_grad = False

    trainable_params = count_trainable_parameters(model)
    print(f"   → Paramètres entraînables: {trainable_params:,}")

    # 3) Prétraitement / tokenisation
    print("\n🔹 Prétraitement des données ...")
    tokenized_dataset = preprocess_dataset(raw_dataset, tokenizer)

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # (Optionnel) : réduire un peu la taille pour un test rapide
    if USE_SUBSET:
        print(f"\n⚠️  Mode SUBSET activé - Réduction dataset pour test rapide")
        train_dataset = train_dataset.select(range(min(TRAIN_SUBSET_SIZE, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(EVAL_SUBSET_SIZE, len(eval_dataset))))
        print(f"   - Train size: {len(train_dataset)}")
        print(f"   - Eval size: {len(eval_dataset)}")

    # 4) Config d'entraînement
    print(f"\n🔹 Configuration de l'entraînement ...")
    print(f"   - Output directory: {OUTPUT_DIR}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Per-device batch size: {PER_DEVICE_BATCH_SIZE}")
    print(f"   - Gradient accumulation steps: {GRAD_ACC_STEPS}")
    print(f"   - Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"   - Learning rate: {LR}")
    print(f"   - Dataset size: {len(train_dataset)} train, {len(eval_dataset)} eval")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=200,  # Log moins fréquemment (optimisé pour 1 epoch)
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        save_total_limit=2,  # Garder seulement les 2 meilleurs checkpoints
        fp16=False,          # On peut passer à True si ça passe bien, pour encore réduire la VRAM
        dataloader_num_workers=0,  # 0 pour éviter problèmes Windows multiprocessing
    )

    # 5) Trainer HF
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) Fine-tuning
    print("\n" + "=" * 80)
    print("🚀 LANCEMENT DU FINE-TUNING")
    print("=" * 80 + "\n")

    trainer.train()

    print("\n✅ Entraînement terminé. Sauvegarde du modèle ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Modèle sauvegardé dans {OUTPUT_DIR}")

    # 7) Évaluation finale sur le test set
    print("\n" + "=" * 80)
    print("📊 ÉVALUATION SUR LE JEU DE TEST")
    print("=" * 80 + "\n")

    test_dataset = tokenized_dataset["test"]
    metrics = trainer.evaluate(test_dataset)

    print("\n📊 Métriques finales (test set):")
    print("-" * 60)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key:.<40} {value:.4f}")
    print("-" * 60)

    # 8) Sauvegarder info d'entraînement
    save_training_info(OUTPUT_DIR, metrics)

    print("\n" + "=" * 80)
    print("✅ FINE-TUNING TERMINÉ AVEC SUCCÈS")
    print("=" * 80)
    print(f"\n📁 Modèle disponible dans: {OUTPUT_DIR}")
    print(f"📊 Logs TensorBoard disponibles dans: {os.path.join(OUTPUT_DIR, 'logs')}")
    print("\nPour utiliser le modèle fine-tuné dans l'application:")
    print(f"  1. Modifier app/config.py")
    print(f"  2. Changer SENTIMENT_MODEL = '{OUTPUT_DIR}'")
    print(f"  3. Redémarrer l'application\n")


if __name__ == "__main__":
    main()
