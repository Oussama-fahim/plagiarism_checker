================================================
Documentation du projet IA de plagiarism checker
================================================

Bienvenue dans la documentation du projet Détection de plagiat par Intelligence Artificielle à l'aide de RAG et d'analyse sémantique avancée.Ce document détaille les méthodologies, les outils utilisés, ainsi que les résultats obtenus pour identifier les cas de plagiat, qu’ils soient exacts, paraphrasés ou sémantiquement similaires.L’objectif de ce projet est de développer un système de détection robuste en combinant plusieurs approches

*Table des matières*

  - introduction
  - objectifs du projet
  - installation
  - pipeline 
  - creation d'une base de donnés vectorielle a partir de llama_parse 
  - application des approches (recherche hybride)
  - visualisation des résultat 
  - création d'une interface streamlit 
  - Travaux Futurs
  - conclusion


.. AI Plagiarism Sentinel Pro documentation master file

==============================================
AI Plagiarism Sentinel Pro – Détection avancée
==============================================

Introduction
============

La détection automatique du plagiat est devenue un enjeu crucial dans le monde académique et professionnel, où la vérification de l'originalité d'un contenu est essentielle à la crédibilité intellectuelle et à l’intégrité des travaux. 

Ce projet propose une solution intelligente et robuste de détection de plagiat en temps réel, exploitant les dernières avancées en **traitement du langage naturel (NLP)**, **recherche vectorielle**, et **modèles de similarité sémantique multilingue**.

En s’appuyant sur une combinaison d’**embeddings puissants**, de **moteurs de recherche hybride**, de **modèles cross-encoders**, ainsi que d’une **analyse stylistique avancée**, cette plateforme permet de détecter différents types de plagiat : copie exacte, paraphrase, similitude conceptuelle, ou correspondance multilingue.

Le tout est encapsulé dans une interface interactive développée avec **Streamlit**, offrant à l'utilisateur une expérience fluide, visuelle, et totalement explicable.

Objectifs du projet
===================

- **Fournir un système de détection de plagiat automatisé et intelligent** :
  
  - Basé sur une base vectorielle *Chroma* enrichie par des embeddings (*Ollama embeddings*).
  - Capable d’identifier non seulement des copies exactes, mais aussi des paraphrases et similitudes conceptuelles, y compris entre langues différentes (français/anglais).

- **Démontrer l’utilisation combinée de technologies modernes** :
  
  - 🧠 *LangChain*, *TF-IDF*, *Cross-Encoder* (MS-MARCO) pour l’analyse sémantique.
  - 📚 *SpaCy* et *textstat* pour l’analyse stylistique, lisibilité, structure grammaticale, diversité lexicale.
  - 📊 *Streamlit*, *Plotly*, *WordCloud*, *Pyvis* pour la visualisation avancée et l’explicabilité.

- **Assurer une analyse complète et interprétable** :
  
  - Génération d’un rapport technique JSON détaillé.
  - Résumé exécutif avec score global, alertes de plagiat, et recommandations.
  - Visualisation des correspondances (réseaux interactifs, nuages de mots, graphiques).

- **Garantir une expérience utilisateur professionnelle** :
  
  - Interface responsive avec personnalisation CSS.
  - Chargement de texte via saisie directe, fichiers (.pdf, .docx…), ou URL.
  - Téléchargement dynamique de rapports avec score et diagnostic.




Installation
============

Les bibliothèques suivantes sont nécessaires pour le projet :

1. **os** : Manipulation des fichiers et chemins.
2. **pickle** : Sérialisation et sauvegarde des objets Python.
3. **cv2** : Traitement d'images avec OpenCV.
4. **numpy** : Manipulation de matrices et calculs numériques.
5. **streamlit** : Interface web interactive pour l’utilisateur.
6. **langchain.vectorstores** : Gestion des bases vectorielles avec Chroma.
7. **langchain_community.embeddings** : Génération des embeddings avec Ollama.
8. **langdetect** : Détection automatique de la langue d’un texte.
9. **typing** : Annotations de types (`List`, `Dict`, etc.).
10. **time** : Gestion du temps (durée d’analyse, timestamps).
11. **pandas** : Manipulation et affichage de tableaux de données.
12. **difflib** : Comparaison de chaînes pour la similarité exacte.
13. **matplotlib.pyplot** : Visualisation de données.
14. **plotly.express** : Graphiques interactifs (camemberts, barres).
15. **collections.defaultdict** : Regroupement d’éléments similaires.
16. **re** : Expressions régulières pour le nettoyage de texte.
17. **hashlib** : Hachage des textes pour la détection exacte.
18. **sentence_transformers** : Calcul avancé de similarité sémantique (CrossEncoder).
19. **sklearn.feature_extraction.text** : TF-IDF vectorisation.
20. **sklearn.metrics.pairwise** : Similarité cosinus.
21. **PIL.Image** : Chargement et affichage d’images.
22. **requests** : Requête HTTP pour charger des images ou contenus.
23. **io.BytesIO** : Manipulation de contenu binaire.
24. **json** : Sérialisation JSON pour les rapports.
25. **networkx** : Création de graphes de similarité.
26. **pyvis.network** : Visualisation interactive de réseaux.
27. **tempfile** : Création de fichiers temporaires.
28. **spacy** : Analyse grammaticale et entités nommées.
29. **wordcloud.WordCloud** : Nuage de mots basé sur les correspondances.
30. **textstat** : Analyse de lisibilité.
31. **streamlit.components.v1.html** : Affichage de composants HTML personnalisés.
32. **docx2txt** : Extraction de texte depuis fichiers Word.
33. **PyPDF2** : Extraction de texte depuis fichiers PDF.
34. **base64** : Encodage d’images pour l’affichage CSS.
35. **annotated_text** : Mise en évidence de texte dans Streamlit.
36. **st_aggrid** : Tableaux interactifs dans Streamlit.
37. **ollama** : Requêtes vers un modèle de langage local.

.. code-block:: python

   import os
   import pickle
   import cv2
   import numpy as np
   import streamlit as st
   import time
   import pandas as pd
   import matplotlib.pyplot as plt
   import plotly.express as px
   import re
   import hashlib
   import json
   import tempfile
   import requests
   import base64
   import docx2txt
   import PyPDF2
   import networkx as nx
   from PIL import Image
   from io import BytesIO
   from difflib import SequenceMatcher
   from collections import defaultdict
   from typing import List, Dict, Any, Tuple
   from sentence_transformers import CrossEncoder
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity
   from langchain.vectorstores import Chroma
   from langchain_community.embeddings import OllamaEmbeddings
   from wordcloud import WordCloud
   import spacy
   import textstat
   from streamlit.components.v1 import html
   from annotated_text import annotated_text
   from st_aggrid import AgGrid
   import ollama


pipeline
========
.. list-table::
   :widths: 100 200
   :align: center

   * - .. image:: image/1.png
         :alt: Image 1
         :width: 700px
     - .. image:: image/2.png
         :alt: Image 2
         :width: 700px

explication de pipline:
-----------------------
Phase 1: Préparation de la Base Vectorielle
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Étape
     - Outils/Méthodes
     - Description
   * - 1. Extraction
     - LlamaParse (FR/EN)
     - Conversion des PDF/DOCX en Markdown propre
   * - 2. Nettoyage
     - Regex + Unicode Normalization
     - Suppression des en-têtes, pieds de page, caractères spéciaux
   * - 3. Découpage
     - ``split('\\n\\n')``
     - Séparation en paragraphes (1 paragraphe = 1 document)
   * - 4. Embeddings
     - OllamaEmbeddings (mxbai-embed-large)
     - Vectorisation des paragraphes
   * - 5. Stockage
     - Chroma DB
     - Indexation avec métadonnées (source, langue)
   * - 6. Persistance
     - ``vecdb.persist()``
     - Sauvegarde locale dans ``philo_db``

Phase 2: Analyse de Plagiat (Frontend/Backend)
----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Étape
     - Outils/Méthodes
     - Description
   * - 1. Input Utilisateur
     - Streamlit (``file_uploader``/``text_area``)
     - Support pour texte direct, fichiers, ou URLs
   * - 2. Pré-processing
     - ``langdetect`` + ``spacy``
     - Détection de langue et nettoyage
   * - 3. Recherche Hybride
     - Combinaison de 3 méthodes:
     - 
   * - 
     - • **Exact Match** (MD5 + ``SequenceMatcher``)
     - Détection de copies mot-à-mot
   * - 
     - • **Semantic Search** (Ollama + Cross-Encoder)
     - Similarité conceptuelle (seuil: 0.4-1.0)
   * - 
     - • **Multilingue** (Traduction via Llama3)
     - Comparaison FR↔EN
   * - 4. Post-Traitement
     - ``networkx`` + ``pyvis``
     - Génération du réseau de similarité
   * - 5. Rapport
     - JSON + Streamlit
     - Export des résultats détaillés


Création d'une Base de Données Vectorielle avec LlamaParse
===========================================================

Ce guide fournit une procédure complète pour transformer un ou plusieurs fichiers PDF en une base de données vectorielle, utilisable notamment pour la détection de similarité textuelle ou de plagiat. L'approche repose sur l'utilisation combinée de **LlamaParse** pour l'extraction intelligente de texte structuré et de **LangChain** pour la vectorisation et la gestion des documents.

.. contents:: Sommaire
   :depth: 2
   :local:

Étape 1 : Installation des Dépendances
--------------------------------------

Cette première étape consiste à importer l'ensemble des librairies nécessaires au bon fonctionnement du pipeline. 

.. code-block:: python
   :linenos:

   import os
   from llama_parse import LlamaParse
   from llama_parse.base import ResultType
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.vectorstores import Chroma
   from langchain.embeddings import HuggingFaceEmbeddings
   from langchain_core.documents import Document
   from llama_cloud_services.parse.utils import Language
   from langchain_community.embeddings.ollama import OllamaEmbeddings

Les modules importés remplissent des rôles spécifiques :
- `llama_parse` permet d'extraire le contenu structuré des PDF (en Markdown ici).
- `langchain` permet de gérer la transformation du texte en vecteurs ainsi que leur stockage dans une base.
- `OllamaEmbeddings` fournit un modèle d'embedding performant pour convertir du texte en vecteurs numériques.

Étape 2 : Configuration de l'API LlamaParse
-------------------------------------------

Avant de lancer l'extraction, il est nécessaire de configurer LlamaParse avec une clé API valide. On peut également spécifier la langue du document pour améliorer la précision de l’analyse.

.. code-block:: python
   :linenos:

   os.environ["LLAMA_CLOUD_API_KEY"] = "llx-a2C7FgYfP1hzX3pXuvtdaNmexAqsuRnJIJ2G6MjbBrfuS3QY"
   
   parser_fr = LlamaParse(
       result_type=ResultType.MD, 
       language=Language.FRENCH
   )
   parser_en = LlamaParse(
       result_type=ResultType.MD,
       language=Language.ENGLISH
   )

Deux parseurs sont initialisés ici : un pour les documents en français et un autre pour ceux en anglais. Le format de sortie sélectionné est le Markdown (`ResultType.MD`), ce qui permet de conserver la structure logique du document original (titres, paragraphes, listes, etc.).

Étape 3 : Extraction du Contenu PDF
-----------------------------------

On procède ensuite à l’extraction effective du contenu des fichiers PDF. LlamaParse utilisant des appels asynchrones, l’environnement doit être adapté pour gérer cela correctement.

.. code-block:: python
   :linenos:

   import nest_asyncio
   nest_asyncio.apply()

   pdf_files = [("philosophie.pdf", parser_fr)]
   
   with open("plagia_data.md", 'w', encoding='utf-8') as f:
       for file_name, parser in pdf_files:
           print(f"Traitement de {file_name}...")
           documents = parser.load_data(file_name)
           f.write(f"# Contenu extrait de : {file_name}\\n\\n")
           for doc in documents:
               f.write(doc.text + "\\n\\n")

Chaque fichier est traité indépendamment. Le texte extrait est structuré et stocké dans un fichier Markdown intermédiaire (`plagia_data.md`). Cela facilite les traitements ultérieurs, notamment pour la segmentation en paragraphes ou sections.

Étape 4 : Préparation des Données
---------------------------------

Une fois le contenu extrait, il est lu depuis le fichier Markdown et segmenté en paragraphes. Ces derniers seront convertis en objets `Document`, reconnus par LangChain.

.. code-block:: python
   :linenos:

   with open("plagia_data.md", encoding='utf-8') as f:
       markdown_content = f.read()
   
   paragraphs = [p.strip() for p in markdown_content.split('\\n\\n') if p.strip()]
   documents = [Document(page_content=paragraph) for paragraph in paragraphs]

Chaque double saut de ligne est interprété comme une séparation logique entre les idées ou blocs de contenu. Cette segmentation est cruciale pour que les embeddings soient cohérents et représentatifs du contenu.

Étape 5 : Génération des Embeddings
-----------------------------------

Cette étape est centrale : elle convertit le texte en vecteurs numériques à l’aide d’un modèle d’embedding compatible avec LangChain. Ces vecteurs sont ensuite stockés dans une base Chroma persistante.

.. code-block:: python
   :linenos:

   embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
   
   vecdb = Chroma.from_documents(
       documents=documents,
       embedding=embeddings,
       persist_directory="philo_db",
       collection_name="rag-chroma"
   )
   vecdb.persist()

Le modèle utilisé ici, `mxbai-embed-large:latest`, encode chaque paragraphe en un vecteur dense de 1024 dimensions. Ces vecteurs sont ensuite indexés et sauvegardés localement dans un dossier nommé `philo_db`. La collection `rag-chroma` permet de regrouper les documents selon un même thème ou usage.

Résultats
---------

À l'issue de ce processus, une base vectorielle est constituée à partir du contenu textuel extrait.

.. code-block:: text

   Opération terminée avec succès:
   - 914 paragraphes traités
   - Base vectorielle sauvegardée dans: philo_db

Cette base peut désormais être utilisée pour la recherche sémantique, la détection de plagiat ou l’implémentation d’un système RAG (Retrieval-Augmented Generation).

Notes Techniques
----------------

- **Format des embeddings** : chaque paragraphe est transformé en un vecteur de 1024 dimensions, ce qui garantit une bonne expressivité sémantique.
- **Taille moyenne des paragraphes** : entre 150 et 300 mots, ce qui est optimal pour les modèles d’embedding modernes.
- **Métadonnées** : il est possible d’ajouter des métadonnées à chaque `Document` (par exemple la langue, l’origine du fichier, la section du document, etc.) pour des filtres ou recherches avancées.

Conclusion
----------

Ce guide constitue une base robuste pour créer une base vectorielle à partir de documents PDF multilingues. Il est facilement extensible pour inclure plus de fichiers, enrichir les métadonnées ou intégrer des systèmes de recherche sémantique avancée.




Travaux Futurs
==============

1. Améliorer les modèles en utilisant plus de données.
2. Étendre la classification pour inclure d'autres comportements (vapoter, boire, etc.).

Conclusion
==========

Ce projet démontre la puissance de *MediaPipe* et *TensorFlow* pour résoudre des problèmes critiques liés à la sécurité et au bien-être. L'intégration de ces outils offre une solution robuste et extensible.
