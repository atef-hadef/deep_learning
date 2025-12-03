# 🚀 Configuration Groq API

## Étapes pour configurer Groq

### 1. Créer un compte Groq
1. Visitez [console.groq.com](https://console.groq.com)
2. Créez un compte gratuit
3. Accédez à la section "API Keys"

### 2. Générer une clé API
1. Cliquez sur "Create API Key"
2. Donnez un nom à votre clé (ex: "Social Media Analyzer")
3. Copiez la clé générée (elle commence par `gsk_...`)

### 3. Configurer le projet
Ouvrez le fichier `.env` et remplacez :
```env
GROQ_API_KEY=your_groq_api_key_here
```

Par votre vraie clé API :
```env
GROQ_API_KEY=gsk_votre_clé_ici
```

### 4. Modèle configuré
Le modèle par défaut est **llama-3.3-70b-versatile** :
```env
GROQ_MODEL=llama-3.3-70b-versatile
```

### Modèles disponibles sur Groq
- `llama-3.3-70b-versatile` - ⭐ Recommandé (rapide, polyvalent)
- `llama-3.1-70b-versatile` - Alternative stable
- `mixtral-8x7b-32768` - Contexte long (32k tokens)
- `gemma2-9b-it` - Léger et rapide

## ✅ Vérifier la configuration

Lancez le test de vérification :
```bash
python test_llm_verification.py
```

Vous devriez voir :
```
✅ Service LLM disponible
ℹ️  Modèle configuré: llama-3.3-70b-versatile
```

## 🎯 Utilisation

Une fois configuré, le bouton **"Résumé LLM"** dans l'interface utilisera automatiquement Groq pour générer des résumés intelligents basés sur les posts affichés.

## 📊 Limites Groq (Plan gratuit)
- **Vitesse** : Ultra rapide (jusqu'à 300 tokens/sec)
- **Requêtes** : Généralement très généreux
- **Modèles** : Accès à plusieurs modèles open-source

## 🔒 Sécurité
⚠️ **Ne jamais commiter votre clé API dans Git !**
Le fichier `.env` est déjà dans `.gitignore`.

## 💡 Avantages de Groq vs OpenRouter
✅ Plus rapide (inférence optimisée)
✅ Gratuit avec limites généreuses
✅ API compatible OpenAI
✅ Modèles open-source (Llama 3.3)
