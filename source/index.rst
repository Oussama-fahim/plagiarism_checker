
AI Plagiarism Sentinel Pro - Documentation
==========================================

Ce document décrit en profondeur les fonctions de l'application Streamlit `main4.py`, avec un accent particulier sur la **recherche hybride multilingue**.

.. contents:: Table des matières
   :depth: 2
   :local:

Fonctions principales
---------------------

1. initialize_system()
~~~~~~~~~~~~~~~~~~~~~~

**But** : Initialiser les modèles d'embeddings (Ollama + TF-IDF) et la base vectorielle Chroma, essentielle pour toute recherche sémantique.

**Ce que la fonction fait** :

.. code-block:: python

   embeddings = OllamaEmbeddings(...)
   vecdb = Chroma(...)
   tfidf_vectorizer = TfidfVectorizer(...)

**Composants clés** :
- OllamaEmbeddings : encode les documents pour la base vectorielle (via mxbai-embed-large)
- Chroma : moteur de recherche vectorielle
- TfidfVectorizer : encode les textes pour comparaison lexicale (fr/en, ngram_range=(1,3))

**Bonnes pratiques** :
- Utilise @st.cache_resource pour éviter de réinitialiser à chaque interaction.
- Barres de progression pour UX utilisateur fluide.


2. check_exact_match()
~~~~~~~~~~~~~~~~~~~~~~

**But** : Détecter des copies exactes ou très proches par hachage + Similarité de SequenceMatcher.

**Fonctionnement** :
1. Normalisation : minuscules, suppression de ponctuation, espaces
2. Hash MD5 du texte
3. Comparaison avec le hash des documents indexés
4. Si aucun hash exact → recherche de segments longs (8 mots consécutifs)
5. Ajout de correspondances partielles si ratio > 0.7

**Algorithmes** :
- SequenceMatcher pour similarité caractère par caractère


3. translate_text()
~~~~~~~~~~~~~~~~~~~

**But** : Traduire automatiquement la requête ou les documents pour permettre une recherche multilingue.

**Utilise** : ollama.chat(model="llama3.1")

**Intelligence** :
- Ne traduit que si texte > 50 caractères
- Gère les erreurs sans bloquer l’interface (st.warning)


4. calculate_similarity()
~~~~~~~~~~~~~~~~~~~~~~~~~

**But** : Combiner similarité lexicale (TF-IDF) et similarité sémantique (CrossEncoder).

**Fonctionnement** :

.. code-block:: python

   cross_score = cross_encoder.predict(...)
   tfidf_sim = cosine_similarity(tfidf_vec1, tfidf_vec2)
   combined = (cross_score * 0.7) + (tfidf_sim * 0.3)

**Pourquoi c’est puissant** :
- TF-IDF capture la proximité lexicale (n-grammes)
- CrossEncoder capture le sens profond et les paraphrases


5. hybrid_search() ⚡ Fonction maîtresse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Effectuer une recherche intelligente multilingue et multi-algorithmes.

**Étapes détaillées** :

1. Détection de langue

.. code-block:: python

   query_lang = detect(query)

2. Vérification de plagiat exact

.. code-block:: python

   exact_matches = check_exact_match(...)

3. Recherche vectorielle sémantique

.. code-block:: python

   vector_results = vecdb.similarity_search_with_score(query)

4. Traduction et recherche croisée

.. code-block:: python

   translated_query = translate_text(query, 'en'/'fr')
   translated_results = vecdb.similarity_search_with_score(translated_query)

5. Score combiné pour chaque document

.. code-block:: python

   calculate_similarity(query, doc_content)

6. Suppression des doublons et tri final

.. code-block:: python

   unique_results = {content: result}

**Pourquoi c’est puissant** :
- Résout le problème des paraphrases entre langues
- Combine exactitude, flexibilité, et sémantique
- Utilise l’intelligence de plusieurs modèles IA


6. analyze_ideas()
~~~~~~~~~~~~~~~~~~

**Objectif** : Identifier les idées similaires même sans formulation identique.

**Méthodologie** :
- On découpe le texte en phrases de plus de 5 mots
- On compare chaque phrase à celles des documents retrouvés
- Si similarity > 0.5, on la considère comme idée similaire
- Regroupement par phrase source


7. process_results()
~~~~~~~~~~~~~~~~~~~~

**Fonction** : Regrouper tous les types de résultats :
- copies exactes
- paraphrases (combined_score > 0.7)
- similarités conceptuelles (<= 0.7)
- analyse d’idées (analyze_ideas())


8. generate_full_report()
~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Créer un rapport JSON complet avec :
- scores
- correspondances
- idées
- style d’écriture


9. Fonctions visuelles & stylistiques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Exemples** :
- display_style_analysis()
- generate_wordcloud()
- create_similarity_network()

Ces fonctions enrichissent l’interprétation et l’analyse humaine avec des éléments visuels impactants.

Conclusion
----------

La fonction ``hybrid_search()`` est le cœur intelligent de l’application :
- combine détection exacte + vectorielle + cross-langue
- permet une robustesse contre le plagiat déguisé ou traduit
- combine modèles puissants (TF-IDF + CrossEncoder + embeddings sémantiques)
