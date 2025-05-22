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
  - création d'une interface streamlit 
  - résultat
  - Travaux Futurs
  - conclusion

.. toctree::
   :maxdepth: 2
   :hidden:
   
   documentation

.. raw:: html

   <div class="sidebar">
   <div class="toctree-wrapper">
   <h2>Table des matières</h2>
   {{ toctree(maxdepth=2) }}
   </div>
   </div>


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
   :widths: 200 200
   :align: center

   * - .. image:: image/1.png
         :alt: PIPLINE 
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



Application des Approches de Recherche Hybride
=============================================

.. contents:: 
   :depth: 3
   :local:

Introduction
------------
La recherche hybride combine plusieurs techniques de similarité textuelle pour détecter le plagiat à différents niveaux :

1. **Recherche exacte** : Détection de copies mot-à-mot
2. **Similarité sémantique** : Identification des paraphrases
3. **Analyse multilingue** : Comparaison entre langues (FR↔EN)

Architecture Principale
----------------------


.. image:: image/3.png
   :alt: architecture de la recherche hybride
   :width: 400px


Fonctions Clés
--------------

check_exact_match()
~~~~~~~~~~~~~~~~~~~
.. code-block:: python
   :linenos:
   :emphasize-lines: 3-5,12-15

   def check_exact_match(input_text: str, dataset: List[str]) -> List[Tuple[str, float]]:
       """Vérifie les correspondances exactes avec normalisation avancée"""
       def normalize(text):
           text = re.sub(r'[^\w\s]', '', text.strip().lower())
           return re.sub(r'\s+', ' ', text)
       
       normalized_input = normalize(input_text)
       input_hash = hashlib.md5(normalized_input.encode('utf-8')).hexdigest()
       
       for doc in dataset:
           normalized_doc = normalize(doc)
           doc_hash = hashlib.md5(normalized_doc.encode('utf-8')).hexdigest()
           
           if input_hash == doc_hash:  # Match exact
               return [(doc, 1.0)]
           # Similarité textuelle avec SequenceMatcher
           match_ratio = SequenceMatcher(None, normalized_input, normalized_doc).ratio()

**Explication** : 

Cette fonction implémente la première couche de la recherche hybride :

1. Normalisation du texte (minuscules, suppression ponctuation)
2. Hashing MD5 pour les correspondances exactes
3. ``SequenceMatcher`` pour les similarités textuelles (>70%)
4. Détection de segments longs (fenêtres de 8 mots)

translate_text()
~~~~~~~~~~~~~~~~
.. code-block:: python
   :linenos:

   @st.cache_data(ttl=3600)
   def translate_text(text: str, target_lang: str) -> str:
       """Traduction intelligente avec gestion des erreurs"""
       try:
           if len(text) < 50:  # Ne pas traduire les textes trop courts
               return text
               
           response = ollama.chat(
               model="llama3.1",
               messages=[{
                   "role": "system",
                   "content": f"Traduis ce texte en {target_lang}..."
               }],
               options={'temperature': 0.1}
           )
           return response["message"]["content"]

**Rôle** :  

Permet la composante multilingue de la recherche hybride :

- Utilise Ollama/Llama3 pour les traductions FR↔EN
- Cache les résultats pour 1 heure (optimisation performance)
- Gère les textes courts (ne traduit pas en dessous de 50 caractères)

calculate_similarity()
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python
   :linenos:

   def calculate_similarity(text1: str, text2: str) -> float:
       """Calcule la similarité combinée TF-IDF + Cross-Encoder"""
       # Similarité lexicale (TF-IDF)
       vectors = tfidf_vectorizer.transform([text1, text2])
       tfidf_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
       
       # Similarité sémantique (Cross-Encoder)
       cross_score = cross_encoder.predict([[text1, text2]])[0]
       
       return (cross_score * 0.7) + (tfidf_sim * 0.3)  # Combinaison pondérée

**Fonctionnement** :  

Coeur de l'approche hybride :

1. **TF-IDF** : Similarité surfacelle (mots-clés, n-grams)
2. **Cross-Encoder** : Compréhension sémantique profonde
3. Pondération : 70% sémantique + 30% lexicale

hybrid_search()
~~~~~~~~~~~~~~~
.. code-block:: python
   :linenos:
   :emphasize-lines: 8-9,15-17,25-27

   def hybrid_search(query: str, dataset: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
       """Recherche hybride multilingue"""
       # 1. Détection langue
       query_lang = detect(query) if len(query) > 20 else 'en'
       
       # 2. Recherche exacte
       exact_matches = check_exact_match(query, dataset)
       if exact_matches:
           return [{"match_type": "exact", ...}]
       
       # 3. Recherche vectorielle
       vector_results = vecdb.similarity_search_with_score(query, k=top_k*2)
       
       # 4. Recherche multilingue
       if query_lang == 'fr':
           translated_query = translate_text(query, 'en')
           translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k)
       
       # Combinaison et filtrage
       all_results = [...]
       return sorted(all_results, key=lambda x: x["combined_score"], reverse=True)[:top_k]

**Workflow** :

1. Orchestre les différentes méthodes de recherche
2. Combine les résultats natifs et traduits
3. Applique des pénalités aux résultats traduits (-10%)
4. Trie par score combiné

analyze_ideas()
~~~~~~~~~~~~~~~
.. code-block:: python
   :linenos:

   def analyze_ideas(input_text: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       """Analyse des similarités conceptuelles entre phrases"""
       ideas = []
       sentences = [s.strip() for s in re.split(r'[.!?]', input_text) if len(s.strip().split()) > 5]
       
       for match in matches:
           if match["combined_score"] < 0.4:  # Seuil minimal
               continue
           match_sentences = [s.strip() for s in re.split(r'[.!?]', match["content"]) if len(s.strip().split()) > 5]
           
           for sent in sentences:
               for match_sent in match_sentences:
                   sim_score = calculate_similarity(sent, match_sent)
                   if sim_score > 0.5:  # Seuil idée similaire
                       ideas.append({
                           "source_sentence": sent,
                           "matched_sentence": match_sent,
                           "similarity": sim_score
                       })

**Objectif** :  
Détecte les plagiat conceptuel en :

- Découpant le texte en phrases
- Comparant chaque paire de phrases
- Gardant les matches >50% de similarité
- Groupant les idées similaires

Visualisation des Résultats
---------------------------

create_similarity_network()
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python
   :linenos:

   def create_similarity_network(matches):
       """Crée un réseau de similarité interactif"""
       G = nx.Graph()
       for i, match in enumerate(matches):
           G.add_node(f"Source_{i}", color='blue')
           G.add_node(match['source'], color='red')
           G.add_edge(f"Source_{i}", match['source'], weight=match['score'])
       
       net = Network(height="500px")
       net.from_nx(G)
       return net

**Rôle** :  
Génère une visualisation interactive des connexions entre :

- Le texte source (nœuds bleus)
- Les documents trouvés (nœuds rouges)
- Les arêtes pondérées par le score de similarité

Conclusion
----------
Cette approche hybride combine :

- **Précision** : Détection des copies exactes
- **Nuance** : Compréhension sémantique
- **Couverure** : Analyse multilingue
- **Transparence** : Visualisations explicatives

création d'une interface streamlit 
==================================

Cette partie détaille la conception et l'implémentation d'une interface Streamlit complète pour une application de détection de plagiat AI-powered.

.. contents:: Table des Matières
   :depth: 3
   :local:

Introduction
------------

L'interface Streamlit a été conçue pour offrir une expérience utilisateur riche avec :

- Un dashboard interactif
- Des visualisations de données avancées
- Une analyse en temps réel
- Un design responsive et moderne


Configuration Initiale
----------------------

.. code-block:: python

    import streamlit as st
    st.set_page_config(
        layout="wide", 
        page_title="🔍 AI Plagiarism Sentinel Pro", 
        page_icon="🔍"
    )

Explications :
~~~~~~~~~~~~~~
- ``layout="wide"`` permet d'utiliser toute la largeur de l'écran
- Personnalisation du titre et de l'icône pour une identité visuelle

Initialisation des Modèles
--------------------------

.. code-block:: python

    @st.cache_resource(show_spinner=False)
    def initialize_system():
        # Initialisation des embeddings
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
            temperature=0.01,
            top_k=50
        )
        
        # Initialisation de la base vectorielle
        vecdb = Chroma(
            persist_directory="philo_db",
            embedding_function=embeddings,
            collection_name="rag-chroma"
        )

Explications :
~~~~~~~~~~~~~~
- ``@st.cache_resource`` optimise les performances en cachant les ressources initialisées
- La fonction charge les modèles NLP et la base de données vectorielle

--------------------
Interface Utilisateur
--------------------

En-tête Personnalisé
--------------------

.. code-block:: python

    def load_assets():
        try:
            response = requests.get("https://images.unsplash.com/photo-1620712943543-bcc4688e7485")
            banner = Image.open(BytesIO(response.content))
            return banner
        except:
            return None

    banner_image = load_assets()
    if banner_image:
        st.image(banner_image, use_column_width=True)
    else:
        st.markdown("""
        <div class="header">
            <h1>🔍 AI Plagiarism Sentinel Pro</h1>
        </div>
        """, unsafe_allow_html=True)

Explications :
~~~~~~~~~~~~~~
- Téléchargement dynamique d'une bannière
- Fallback sur un en-tête HTML si l'image n'est pas disponible

Sidebar Configurable
--------------------

.. code-block:: python

    with st.sidebar:
        st.title("⚙️ Paramètres Experts")
        
        with st.expander("🔍 Options de Recherche", expanded=True):
            analysis_mode = st.selectbox(
                "Mode d'analyse",
                ["DeepScan Pro", "Rapide", "Manuel Expert"]
            )
            
            sensitivity = st.slider(
                "Niveau de sensibilité",
                1, 10, 8
            )

Explications :
~~~~~~~~~~~~~~
- Organisation des contrôles dans des expanders
- Utilisation de widgets Streamlit variés (selectbox, slider)

Zone de Saisie Multimode
------------------------

.. code-block:: python

    input_method = st.radio(
        "Source d'entrée",
        ["📝 Texte direct", "📂 Fichier", "🌐 URL"],
        horizontal=True
    )
    
    if input_method == "📂 Fichier":
        uploaded_file = st.file_uploader(
            "Téléversez un document",
            type=["txt", "pdf", "docx"]
        )

Explications :
~~~~~~~~~~~~~~
- Interface unifiée pour différentes méthodes de saisie
- Traitement spécifique pour chaque type d'entrée

Visualisations Avancées
-----------------------

Cartes de Résultats
-------------------

.. code-block:: python

    def display_match_card(match):
        with st.container():
            st.markdown(f"""
            <div class="{'exact-match' if match['match_type'] == 'exact' else 'partial-match'}">
                <h3>{match['match_type'].capitalize()} - Score: {match['combined_score']*100:.1f}%</h3>
                <p><strong>Source:</strong> {match['metadata'].get('source', 'Inconnue')}</p>
            </div>
            """, unsafe_allow_html=True)

Explications :
~~~~~~~~~~~~~~
- Utilisation de HTML/CSS pour des cartes stylisées
- Classes CSS dynamiques en fonction du type de correspondance

Réseau de Similarité
--------------------

.. code-block:: python

    def create_similarity_network(matches):
        G = nx.Graph()
        for i, match in enumerate(matches):
            G.add_node(f"Source_{i}", size=15, color='blue')
            G.add_edge(f"Source_{i}", match['metadata'].get('source', f"Doc_{i}"), 
                      weight=match['combined_score'])
        
        net = Network(height="500px", width="100%")
        net.from_nx(G)
        return net

Explications :
~~~~~~~~~~~~~~
- Utilisation de NetworkX pour la création du graphe
- Intégration avec PyVis pour le rendu interactif

---------------------
Gestion des Données
-------------------

Cache et Performance
--------------------

.. code-block:: python

    @st.cache_data(ttl=3600)
    def translate_text(text: str, target_lang: str) -> str:
        # Fonction de traduction
        return translated_text

Explications :
~~~~~~~~~~~~~~
- ``@st.cache_data`` pour cacher les résultats coûteux
- TTL (Time-To-Live) de 1 heure pour les traductions

Traitement des Fichiers
-----------------------

.. code-block:: python

    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(uploaded_file)

Explications :
~~~~~~~~~~~~~~
- Support multi-format (PDF, DOCX, etc.)
- Extraction robuste du texte

------------------
Design Avancé
-------------

CSS Personnalisé
----------------

.. code-block:: python

    def apply_custom_css():
        css = """
        <style>
            .header {
                background: linear-gradient(135deg, #434343 0%, #000000 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
            }
            .exact-match { 
                border-left: 6px solid #ef4444; 
                background-color: rgba(239, 68, 68, 0.05);
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

Explications :
~~~~~~~~~~~~~~
- Styles CSS intégrés directement dans Streamlit
- Utilisation de gradients et d'effets modernes

Mise en Page
------------

.. code-block:: python

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score maximal", f"{score:.1f}%")
    with col2:
        st.metric("Correspondances", count)

Explications :
~~~~~~~~~~~~~~
- Layout multi-colonnes pour une organisation optimale
- Widgets de métriques pour les KPI

--------------------
Fonctionnalités Avancées
------------------------

Onglets Interactifs
-------------------

.. code-block:: python

    tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Correspondances"])
    with tab1:
        st.plotly_chart(fig)
    with tab2:
        for match in matches:
            display_match_card(match)

Explications :
~~~~~~~~~~~~~~
- Navigation par onglets pour organiser le contenu
- Contenu dynamique dans chaque onglet

Génération de Rapports
----------------------

.. code-block:: python

    def generate_full_report(results):
        return json.dumps({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_matches": len(results.get('all_matches', []))
            },
            "results": {
                "highest_score": results.get('highest_score', 0),
            }
        }, indent=2)

Explications :
~~~~~~~~~~~~~~
- Format JSON structuré
- Téléchargement direct via Streamlit

--------------------
Conclusion
----------

Cette interface Streamlit combine :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Des composants UI riches
- Des visualisations interactives
- Une gestion efficace des données
- Un design moderne personnalisable

Les techniques présentées peuvent être adaptées pour tout type d'application data-centric.

resultas
========



Travaux futurs
==============

Cette partie présente les améliorations potentielles pour la future version du système de détection de plagiat.

1. Améliorations des Algorithmes
--------------------------------

1.1. Intégration de Modèles Multilingues Avancés
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Ajout de modèles spécialisés pour d'autres langues (espagnol, allemand, chinois)
- Implémentation d'un système de détection automatique de langue plus robuste
- Optimisation des traductions avec des modèles dédiés (NLLB, DeepL)

1.2. Amélioration des Scores de Similarité
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Combinaison de plusieurs métriques (BERTScore, ROUGE, BLEU)
- Ajout d'un système de pondération dynamique basé sur le contexte
- Intégration de modèles de similarité spécifiques aux domaines (scientifique, juridique)

2. Fonctionnalités Avancées
---------------------------

2.1. Analyse Temporelle
~~~~~~~~~~~~~~~~~~~~~~~
- Détection des variations stylistiques dans le texte
- Identification des ajouts/modifications successifs
- Reconstruction de l'historique d'écriture

2.2. Détection de Paraphrase Sophistiquée
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Modèles spécifiques pour identifier les paraphrases avancées
- Détection des modifications structurelles (changement d'ordre des idées)
- Analyse des patterns de réécriture

3. Interface Utilisateur
------------------------

3.1. Tableau de Bord Analytique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Visualisations interactives des résultats
- Comparaison avec les soumissions précédentes
- Suivi des améliorations dans les révisions

3.2. Outils d'Aide à la Réécriture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Suggestions de reformulation originales
- Générateur de citations automatiques
- Identification des passages à risque

4. Infrastructure Technique
---------------------------

4.1. Optimisation des Performances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Implémentation d'un système de cache distribué
- Prétraitement asynchrone des documents
- Indexation incrémentielle

4.2. Extension des Bases de Référence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Intégration de nouvelles sources académiques
- Connexion aux bases de données ouvertes
- Mise à jour automatique du corpus de référence

5. Intégrations Système
-----------------------

5.1. API Universelle
~~~~~~~~~~~~~~~~~~~~
- Développement d'une API RESTful complète
- Intégration avec les LMS (Moodle, Canvas)
- Connecteurs pour les outils d'édition (Word, Google Docs)

5.2. Modules Spécialisés
~~~~~~~~~~~~~~~~~~~~~~~~
- Version pour l'édition scientifique
- Module dédié à l'éducation
- Solution pour les éditeurs professionnels

Perspectives à Long Terme
-------------------------

- Analyse multimodale (texte + images + formules)
- Détection cross-média (vidéos, podcasts)
- Système prédictif de risque de plagiat
- Blockchain pour la traçabilité des sources

Ces améliorations permettront de positionner l'outil comme une solution complète de vérification d'intégrité académique et professionnelle.


Conclusion
==========

Après la réalisation de ce projet AI Plagiarism Sentinel Pro, plusieurs constats importants peuvent être tirés :

1. **Efficacité de détection** : 
   Le système combine avec succès différentes approches (correspondance exacte, analyse sémantique, similarité conceptuelle) pour offrir une détection de plagiat multi-niveaux très performante.

2. **Innovation technologique** :
   L'utilisation combinée de modèles de langue (Ollama), d'embeddings vectoriels et de techniques traditionnelles (TF-IDF) permet une analyse à la fois profonde et rapide.

3. **Polyvalence linguistique** :
   La capacité à traiter plusieurs langues (notamment français et anglais) et à identifier des similarités translinguistiques constitue un atout majeur.

4. **Analyse stylistique** :
   Les fonctionnalités d'analyse d'écriture vont au-delà de la simple détection de plagiat, offrant des insights précieux sur le style et la qualité rédactionnelle.

5. **Interface intuitive** :
   Le dashboard Streamlit propose une expérience utilisateur riche tout en restant accessible, avec des visualisations claires et des rapports détaillés.


