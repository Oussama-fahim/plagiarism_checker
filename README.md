# 🔍 AI Plagiarism Sentinel Pro
## 📌 Table des Matières
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Architecture Technique](#-architecture-technique)
- [Structure des Fichiers](#-structure-des-fichiers)
- [Métriques Clés](#-métriques-clés)
- [Cas d'Usage](#-cas-dusage)
- [Benchmarks](#-benchmarks)
- [Limites Connues](#-limites-connues)
- [Licence](#-licence)
- [Contact](#-contact)

## 🚀 Fonctionnalités

### 🔍 Détection Multi-Niveaux
- **Correspondances exactes** (copier-coller)
- **Paraphrases et reformulations**
- **Similarités conceptuelles**
- **Détection multilingue** (FR/EN)

### 📊 Analyse Approfondie
- Score de plagiat avec seuils configurables
- Cartographie des similarités
- Profil stylistique complet
- Nuage de mots clés

### 🖥️ Interface
- Dashboard interactif
- Rapport détaillé exportable (JSON)
- Filtres intelligents
- Support multi-sources

## 🛠️ Installation

### Prérequis
- Python 3.10+
- Ollama ([guide d'installation](https://ollama.ai/))
- Modèles SpaCy :
```bash
python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_sm
