#!/usr/bin/env python3
"""
setup_training.py

Script d'installation et de vérification pour le module de training.
Vérifie les dépendances, télécharge les modèles nécessaires, et prépare l'environnement.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path


def print_header(text):
    """Affiche un header formaté."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_step(step_num, text):
    """Affiche une étape numérotée."""
    print(f"[{step_num}] {text}")


def check_package(package_name, import_name=None):
    """Vérifie si un package est installé."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"  ✅ {package_name}")
        return True
    except ImportError:
        print(f"  ❌ {package_name} - NON INSTALLÉ")
        return False


def install_package(package):
    """Installe un package via pip."""
    print(f"  📦 Installation de {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"  ✅ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError:
        print(f"  ❌ Échec de l'installation de {package}")
        return False


def check_cuda():
    """Vérifie la disponibilité de CUDA."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✅ CUDA disponible")
            print(f"     Device: {torch.cuda.get_device_name(0)}")
            print(f"     Version: {torch.version.cuda}")
        else:
            print(f"  ⚠️  CUDA non disponible (entraînement sur CPU)")
        return cuda_available
    except ImportError:
        print(f"  ❌ PyTorch non installé")
        return False


def check_disk_space():
    """Vérifie l'espace disque disponible."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)  # Convertir en GB
        
        print(f"  💾 Espace disque libre: {free_gb} GB")
        
        if free_gb < 5:
            print(f"  ⚠️  WARNING: Moins de 5 GB disponibles (5 GB recommandés)")
            return False
        else:
            print(f"  ✅ Espace suffisant")
            return True
    except Exception as e:
        print(f"  ⚠️  Impossible de vérifier l'espace disque: {e}")
        return True


def create_directories():
    """Crée les dossiers nécessaires."""
    directories = [
        "models",
        "training",
        "logs",
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Créé: {directory}/")
        else:
            print(f"  ✓  Existe: {directory}/")


def download_spacy_model():
    """Télécharge le modèle spaCy si nécessaire."""
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print(f"  ✅ Modèle spaCy déjà installé")
            return True
        except OSError:
            print(f"  📦 Téléchargement du modèle spaCy...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            print(f"  ✅ Modèle spaCy installé")
            return True
    except ImportError:
        print(f"  ❌ spaCy non installé")
        return False


def test_dataset_download():
    """Teste le téléchargement du dataset."""
    try:
        from datasets import load_dataset
        print(f"  📡 Test de connexion Hugging Face...")
        
        # Essayer de charger juste la config (rapide)
        dataset_info = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train[:1]")
        print(f"  ✅ Connexion Hugging Face OK")
        return True
    except Exception as e:
        print(f"  ❌ Erreur de connexion: {e}")
        return False


def display_summary(results):
    """Affiche un résumé des vérifications."""
    print_header("RÉSUMÉ")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("✅ Tous les tests sont passés avec succès!")
        print("\n🚀 Vous pouvez maintenant lancer le fine-tuning:")
        print("\n    python -m training.train_sentiment_roberta\n")
    else:
        print("⚠️  Certains problèmes ont été détectés:\n")
        for check, status in results.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check}")
        
        print("\n📖 Consultez training/GUIDE_FINETUNING.md pour plus d'aide")


def main():
    """Fonction principale."""
    print_header("SETUP TRAINING MODULE - Vérification Environnement")
    
    results = {}
    
    # 1. Vérifier packages Python
    print_step(1, "Vérification des packages Python")
    required_packages = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "scikit-learn": "sklearn",
        "spacy": "spacy",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    results["Packages Python"] = len(missing_packages) == 0
    
    # Proposer installation des packages manquants
    if missing_packages:
        print(f"\n⚠️  Packages manquants: {', '.join(missing_packages)}")
        response = input("\nVoulez-vous les installer maintenant? (o/n): ")
        if response.lower() in ['o', 'y', 'oui', 'yes']:
            for package in missing_packages:
                install_package(package)
    
    # 2. Vérifier CUDA
    print(f"\n")
    print_step(2, "Vérification GPU/CUDA")
    results["CUDA"] = check_cuda()
    
    # 3. Vérifier espace disque
    print(f"\n")
    print_step(3, "Vérification espace disque")
    results["Espace disque"] = check_disk_space()
    
    # 4. Créer dossiers
    print(f"\n")
    print_step(4, "Création des dossiers")
    create_directories()
    results["Dossiers"] = True
    
    # 5. Modèle spaCy
    print(f"\n")
    print_step(5, "Vérification modèle spaCy")
    results["spaCy model"] = download_spacy_model()
    
    # 6. Test connexion Hugging Face
    print(f"\n")
    print_step(6, "Test connexion Hugging Face")
    results["Hugging Face"] = test_dataset_download()
    
    # Résumé
    display_summary(results)


if __name__ == "__main__":
    main()
