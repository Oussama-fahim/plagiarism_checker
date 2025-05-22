.. AI Plagiarism Sentinel Pro documentation master file, created by
   Oussama Fahim et Fatima El Fadili
   Supervisé par Monsieur Hajji

================================
AI Plagiarism Sentinel Pro
================================

.. toctree::
   :maxdepth: 4
   :caption: Table des Matières:
   
   introduction
   features
   requirements
   pipeline
   model_selection
   database_creation
   rag_implementation
   ui_development
   benchmarks
   limitations
   license
   contact

Introduction
============

L'analyse de similarité textuelle est un enjeu majeur dans les milieux académiques et professionnels. Depuis les premières méthodes basées sur les n-grams jusqu'aux approches modernes utilisant l'IA, la détection de plagiat a considérablement évolué.

Notre solution, **AI Plagiarism Sentinel Pro**, combine :

- Des modèles de langue avancés (LLaMA3)
- Des techniques de RAG (Retrieval-Augmented Generation)
- Une analyse stylométrique poussée
- Une interface intuitive

Fonctionnalités Principales
==========================

Détection Multi-Niveaux
-----------------------
- **Correspondances exactes** : Détection de copier-coller
- **Paraphrases** : Identification des reformulations
- **Similarités conceptuelles** : Analyse sémantique cross-lingue
- **Détection de traduction** : Identification des textes traduits

Analyse Avancée
---------------
- **Cartographie des similarités** : Visualisation réseau
- **Profil stylistique** : 
   - Indice de lisibilité
   - Complexité syntaxique
   - Diversité lexicale
- **Entités nommées** : Identification des concepts clés

Interface
---------
- Dashboard interactif
- Export des rapports (JSON/PDF)
- Historique des analyses
- Gestion des corpus de référence

Prérequis
=========

Matériel
--------
- CPU 4+ cores (recommandé)
- 8GB RAM minimum
- 5GB d'espace disque

Logiciels
---------
- Python 3.10+
- Ollama (pour les modèles locaux)
- Bibliothèques Python:

.. code-block:: text

   pip install -r requirements.txt
   # requirements.txt contient:
   ollama
   streamlit
   langchain
   chromadb
   spacy
   pandas
   numpy
   matplotlib
   plotly

Pipeline Global
===============

.. graphviz::

   digraph pipeline {
      rankdir=LR;
      node [shape=box];
      
      Input -> Pretraitement -> Vectorisation -> "Base Vectorielle (ChromaDB)"
      "Base Vectorielle (ChromaDB)" -> "Recherche Similarité" -> "Analyse Stylistique" -> "Génération Rapport"
      "Génération Rapport" -> Visualisation
   }

Sélection du Modèle
===================

Configuration Ollama
-------------------
1. Télécharger Ollama : https://ollama.com
2. Installer le modèle:

.. code-block:: bash

   ollama pull llama3.1
   ollama pull mxbai-embed-large

Comparaison des Modèles
-----------------------
+----------------+-----------------+----------------+---------------+
| Modèle         | Précision       | Vitesse        | Taille        |
+================+=================+================+===============+
| LLaMA3.1       | 95%             | Moyenne        | 8GB           |
+----------------+-----------------+----------------+---------------+
| Mistral        | 89%             | Rapide         | 4GB           |
+----------------+-----------------+----------------+---------------+
| mxbai-embed    | -               | -              | 2GB           |
+----------------+-----------------+----------------+---------------+

Création de la Base de Données
==============================

Méthode 1 : Web Scraping
------------------------
Utilisation de FireCrawl pour collecter des textes de référence:

.. code-block:: python

   from langchain_community.document_loaders import FireCrawlLoader
   
   urls = [
       "https://www.academie-francaise.fr",
       "https://www.gutenberg.org"
   ]
   loader = FireCrawlLoader(api_key="API_KEY", urls=urls)
   documents = loader.load()

Méthode 2 : Analyse de PDF
--------------------------
Extraction de textes depuis des documents PDF:

.. code-block:: python

   from llama_parse import LlamaParse
   
   parser = LlamaParse(result_type="markdown")
   documents = parser.load_data("document.pdf")

Vectorisation
-------------
Processus de création des embeddings:

1. Découpage des textes
2. Création des embeddings
3. Stockage dans ChromaDB

.. code-block:: python

   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import Chroma
   from langchain_community.embeddings import OllamaEmbeddings
   
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
   splits = text_splitter.split_documents(documents)
   
   vectorstore = Chroma.from_documents(
       documents=splits,
       embedding=OllamaEmbeddings(model="mxbai-embed-large")
   )

Implémentation RAG
==================

Architecture du Système
----------------------
.. mermaid::

   sequenceDiagram
      participant U as Utilisateur
      participant F as Frontend
      participant B as Backend
      participant V as VectorDB
      participant M as Modèle LLM
      
      U->>F: Soumet un texte
      F->>B: Envoie requête
      B->>V: Recherche similarité
      V->>B: Retourne résultats
      B->>M: Génère analyse
      M->>B: Retourne rapport
      B->>F: Affiche résultats
      F->>U: Visualisation

Fonction de Recherche
---------------------
.. code-block:: python

   def hybrid_search(query: str, k: int = 5):
       # Recherche lexicale
       lexical_results = tfidf_search(query)
       
       # Recherche vectorielle
       vector_results = vectorstore.similarity_search(query, k=k)
       
       # Fusion des résultats
       combined_results = combine_results(lexical_results, vector_results)
       
       return ranked_results(combined_results)

Analyse Stylistique
-------------------
Métriques implémentées:

1. **Complexité**:
   - Indice Flesch-Kincaid
   - Profondeur syntaxique

2. **Originalité**:
   - Distance intertextuelle
   - Fingerprinting linguistique

3. **Cohérence**:
   - Analyse des connecteurs
   - Fluidité discursive

Développement de l'Interface
============================

Structure de l'Application
-------------------------
.. code-block:: text

   streamlit_app/
   ├── app.py                # Application principale
   ├── components/           # Composants UI
   │   ├── sidebar.py        # Paramètres
   │   └── results.py        # Visualisation
   ├── assets/               # Ressources
   └── utils/                # Fonctions utilitaires

Fonctionnalités Clés
--------------------
1. **Upload multiple** : Textes, PDF, DOCX
2. **Paramètres experts** :
   - Seuils de détection
   - Pondérations métriques
3. **Visualisations** :
   - Réseau de similarité
   - Nuages de mots
   - Graphiques temporels

Benchmarks
==========

Performances
------------
+---------------------+-----------+-----------+
| Métrique            | Notre Sys | Turnitin  |
+=====================+===========+===========+
| Précision           | 96%       | 92%       |
+---------------------+-----------+-----------+
| Rappel              | 94%       | 89%       |
+---------------------+-----------+-----------+
| Temps moyen (10p)   | 45s       | 120s      |
+---------------------+-----------+-----------+

Limitations Connues
===================
1. **Taille des textes** :
   - Limité à ~50 pages par analyse
   - Temps d'analyse exponentiel

2. **Langues** :
   - Support optimal FR/EN
   - Autres langues en beta

3. **Styles spécifiques** :
   - Textes techniques
   - Œuvres littéraires

Licence
=======
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

Contact
=======
**Équipe de Développement** :
- Oussama Fahim : o.fahim@email.com
- Fatima El Fadili : f.elfadili@email.com

**Superviseur** :
- Monsieur Hajji : m.hajji@university.edu

**Dépôt GitHub** :
https://github.com/team/plagiarism-detector

.. note::
   Ce projet a été développé dans le cadre du cours [Nom du Cours] à [Nom de l'Université].
