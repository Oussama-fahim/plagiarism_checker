# 🛡️ AI Plagiarism Sentinel Pro

Une application de détection de plagiat en temps réel, alimentée par des modèles avancés de traitement du langage naturel (NLP) et des techniques de recherche vectorielle. Elle permet de détecter les copies exactes, les paraphrases, les similarités sémantiques et les idées similaires dans des textes en français et en anglais.

---

## 📂 Table des Matières

* [🌟 Introduction](#🌟-introduction)
* [✨ Fonctionnalités](#✨-fonctionnalités)
* [🧰 Prérequis Techniques](#🧰-prérequis-techniques)
* [📁 Structure du Projet](#📁-structure-du-projet)
* [🚀 Lancement de l’Application](#🚀-lancement-de-lapplication)
* [🧠 Architecture et Fonctionnement](#🧠-architecture-et-fonctionnement)
* [📊 Détails Techniques](#📊-détails-techniques)
* [📂 Jeux de Données](#📂-jeux-de-données)
* [📊 Améliorations Futures](#📊-améliorations-futures)

---

## 🌟 Introduction

**AI Plagiarism Sentinel Pro** est une application IA multi-niveaux dédiée à la détection de plagiat à fort impact visuel et analytique. Elle exploite des embeddings, des modèles de recherche vectorielle, la traduction automatique, et l’analyse stylistique pour repérer des cas de réutilisation de contenu, même dans des formes modifiées ou traduites.

Cas d’utilisation : universités, enseignants, chercheurs, éditeurs, correcteurs IA, journalistes, etc.

---

## ✨ Fonctionnalités

### 🔍 Détection Multiforme de Plagiat

* ✅ Correspondances exactes (copie mot à mot)
* ✅ Similarités sémantiques et idées proches (paraphrases)
* ✅ Traductions croisées FR/EN avec recherche vectorielle multilingue

### 📊 Analyse Stylisitique

* Lisibilité (Flesch, Dale-Chall)
* Diversité lexicale et complexité grammaticale
* Détection des entités nommées (NER)

### 📊 Visualisations

* Graphes interactifs de similarité (PyVis)
* Nuages de mots (WordCloud)
* Histogrammes, bar charts, courbes

### 🔢 Reporting

* Rapport JSON complet
* Export interactif via Streamlit

### 🔐 Interface

* Interface web responsive avec Streamlit
* CSS personnalisé, progress bars, alertes

---

## 🧰 Prérequis Techniques

### 💻 Environnement

```bash
Python ≥ 3.8
```

### 📊 Bibliothèques Python

```txt
streamlit
langchain
sentence-transformers
ollama
spacy
docx2txt
textstat
scikit-learn
PyPDF2
wordcloud
matplotlib
plotly
st-aggrid
networkx
pyvis
langdetect
Pillow
```

### 🛠️ Modèles NLP

```bash
python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_sm
```

---

## 📁 Structure du Projet

```
plagiarismchecker/
├── main4.py               # Application Streamlit principale
├── Untitled-1.ipynb       # Notebook de test et prétraitement
├── philo_db/              # Données vectorielles persistées (Chroma)
├── venv/                  # Ou il ya les modèles et les bibliothèques installés
├── requirements.txt       # Liste complète des dépendances
├── plagiarism.md       # markdown qui était comme origine un pdf
```

---

## 🚀 Lancement de l’Application

### 📦 Installation des dépendances

```bash
pip install -r requirements.txt
```

### 📕 Démarrage

```bash
streamlit run main4.py
```

---

## 🧠 Architecture et Fonctionnement

### 1. 📹 Entrées possibles

* Texte direct
* Fichier PDF / DOCX / TXT
* URL Web

### 2. 🌍 Détection de Langue

* Utilisation de `langdetect`

### 3. ⚖️ Recherche Sémantique

* Embeddings (Ollama + mxbai)
* Recherche vectorielle (Chroma)
* Traduction croissante avec `llama3.1`

### 4. 🎯 Matching

* Comparaison exacte via hashing MD5
* TF-IDF + CrossEncoder pour scoring hybride
* Paraphrases et idées similaires analysées phrase par phrase

### 5. 📊 Visualisation + Rapport

* Diagrammes, score cards, rapport JSON, recommandation finale

---

## 📊 Détails Techniques

* **Recherche vectorielle** : ChromaDB avec embeddings `mxbai-embed-large`
* **Matching exact** : hachage MD5 + `SequenceMatcher`
* **Matching sémantique** : `TF-IDF` + `CrossEncoder` (MiniLM L6 v2)
* **Traduction multilingue** : `ollama.chat` avec `llama3.1`
* **Analyse stylistique** : `spaCy` + `textstat`

---

## 📂 Jeux de Données

* **Corpus vectoriel** : Documents indexés dans `philo_db`
* **Sources** : Documents académiques, essais philosophiques, articles scientifiques (peuvent être personnalisés)
* **Formats pris en charge** : `.txt`, `.pdf`, `.docx`, `.odt`, `.pptx`

---

## 📊 Améliorations Futures

* 🔍 OCR pour documents scannés ou manuscrits
* 📡 API REST pour intégration LMS / Moodle
* 📲 Application mobile simplifiée
* 📈 Analyse comparative multi-documents
* 🗒️ Export PDF du rapport d’analyse

---

## 🔧 Résumé

> AI Plagiarism Sentinel Pro ne se limite pas à repérer les plagiats mot à mot. Il évalue **la forme, le fond, les idées** et même **la stylistique de rédaction** pour une vérification ultra-complète.

Un outil indispensable pour qui prend au sérieux l’intégrité intellectuelle et l’originalité de l’écriture.

---
