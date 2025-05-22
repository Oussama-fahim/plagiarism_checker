================================================
Documentation du projet IA de plagiarism checker
================================================

Bienvenue dans la documentation du projet Détection de plagiat par Intelligence Artificielle à l'aide de RAG et d'analyse sémantique avancée.Ce document détaille les méthodologies, les outils utilisés, ainsi que les résultats obtenus pour identifier les cas de plagiat, qu’ils soient exacts, paraphrasés ou sémantiquement similaires.L’objectif de ce projet est de développer un système de détection robuste en combinant plusieurs approches

*Table des matières*

  - introduction
  - objectifs du projet
  - installation
  - pipeline 
  - creation d'une base de donnés a partir de llama_parse 
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
   :widths: 50 50
   :align: center

   * - .. image:: image/1.png
         :alt: Image 1
         :width: 300px
     - .. image:: image/2.png
         :alt: Image 2
         :width: 300px




Travaux Futurs
==============

1. Améliorer les modèles en utilisant plus de données.
2. Étendre la classification pour inclure d'autres comportements (vapoter, boire, etc.).

Conclusion
==========

Ce projet démontre la puissance de *MediaPipe* et *TensorFlow* pour résoudre des problèmes critiques liés à la sécurité et au bien-être. L'intégration de ces outils offre une solution robuste et extensible.
