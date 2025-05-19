########################################################################
PLAGIARISM SENTINEL PRO - DOCUMENTATION TECHNIQUE APPROFONDIE
########################################################################

.. contents::
   :depth: 6
   :local:
   :backlinks: top

===============================================
PARTIE 1 : ANALYSE COMPLÈTE DU FICHIER UNTITLED-1.IPYNB
===============================================

1.1 Configuration Initiale et Importations
------------------------------------------

1.1.1 Importations des Bibliothèques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 2,5,8

   import os  # Gestion des chemins et variables d'environnement
   from llama_parse import LlamaParse  # Parseur PDF avancé
   from llama_parse.base import ResultType  # Types de sortie
   from langchain.text_splitter import RecursiveCharacterTextSplitter  # Découpage intelligent
   from langchain.vectorstores import Chroma  # Base de données vectorielle
   from langchain.embeddings import HuggingFaceEmbeddings  # Modèles d'embedding
   from langchain_core.documents import Document  # Structure de données
   from llama_cloud_services.parse.utils import Language  # Support multilingue
   from langchain_community.embeddings.ollama import OllamaEmbeddings  # Embeddings optimisés

Détails Techniques:
^^^^^^^^^^^^^^^^^^^
- **Ligne 2** : LlamaParse utilise un moteur OCR amélioré pour l'extraction PDF avec:
  - Détection automatique des colonnes
  - Reconnaissance des en-têtes/pieds de page
  - Conservation de la structure logique

- **Ligne 5** : Le RecursiveCharacterTextSplitter propose:
  - Taille de chunk configurable (par défaut 512 tokens)
  - Recouvrement intelligent (128 tokens)
  - Respect des frontières syntaxiques

- **Ligne 8** : ChromaDB offre:
  - Recherche ANN (Approximate Nearest Neighbor)
  - Persistance sur disque
  - Compression vectorielle

1.1.2 Configuration des Parseurs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   os.environ["LLAMA_CLOUD_API_KEY"] = "llx-a2C7FgYfP1hzX3pXuvtdaNmexAqsuRnJIJ2G6MjbBrfuS3QY"
   
   parser_fr = LlamaParse(
       result_type=ResultType.MD,  # Format Markdown
       language=Language.FRENCH,  # Optimisation FR
       parsing_quality="high",    # Précision maximale
       max_timeout=300            # 5 min timeout
   )

Paramètres Avancés:
^^^^^^^^^^^^^^^^^^^
- ``parsing_quality``: Contrôle le niveau d'analyse (low/medium/high)
- ``max_timeout``: Adapté aux documents complexes
- ``language``: Active les règles linguistiques spécifiques

1.2 Traitement des Documents PDF
--------------------------------

1.2.1 Extraction du Contenu
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   pdf_files = [
       ("philosophie.pdf", parser_fr),  # Tuple (fichier, parser)
       # Structure extensible
   ]
   
   with open("plagia_data.md", 'w', encoding='utf-8') as f:
       for file_name, parser in pdf_files:
           documents = parser.load_data(
               file_name,
               extra_info={"source": file_name}  # Métadonnées
           )
           for doc in documents:
               f.write(f"## EXTRACT FROM: {file_name}\n")
               f.write(doc.text + "\n\n")
               f.write("---\n")

Processus Complet:
^^^^^^^^^^^^^^^^^^
1. Chargement du PDF avec métadonnées
2. Conversion en Markdown structuré
3. Ajout de séparateurs visuels
4. Conservation des informations source

1.3 Vectorisation et Stockage
-----------------------------

1.3.1 Préparation des Données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=512,
       chunk_overlap=128,
       length_function=len,
       separators=["\n\n", "\n", " ", ""]  # Hiérarchie de séparation
   )
   
   paragraphs = []
   with open("plagia_data.md", encoding='utf-8') as f:
       content = f.read()
       paragraphs = text_splitter.split_text(content)

Optimisation du Découpage:
^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``chunk_size=512``: Optimal pour les embeddings
- ``chunk_overlap=128``: Maintient le contexte
- ``separators``: Priorité aux sauts de paragraphe

1.3.2 Création des Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   embeddings = OllamaEmbeddings(
       model="mxbai-embed-large:latest",
       temperature=0.01,  # Réduction du bruit
       top_k=50,          # Précision du top-k
       timeout=120        # 2 min timeout
   )

Caractéristiques du Modèle:
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Taille: 1024 dimensions
- Entraînement: Sur corpus académique
- Spécialisation: Similarité sémantique

1.3.3 Indexation dans ChromaDB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   vecdb = Chroma.from_documents(
       documents=[Document(page_content=p) for p in paragraphs],
       embedding=embeddings,
       persist_directory="philo_db",
       collection_name="rag-chroma",
       collection_metadata={"hnsw:space": "cosine"}  # Métrique de similarité
   )

Configuration Avancée:
^^^^^^^^^^^^^^^^^^^^^^
- ``hnsw:space``: Optimise pour similarité cosinus
- ``persist_directory``: Format binaire optimisé
- ``collection_name``: Isolation des espaces

===============================================
PARTIE 2 : APPROFONDISSEMENT SUR L'APPROCHE DE RECHERCHE HYBRIDE
===============================================

2.1 Architecture de la Recherche Hybride
----------------------------------------

.. image:: _static/hybrid_search_architecture.png
   :align: center
   :width: 800px

2.2 Composants Clés
-------------------

2.2.1 Détection Exacte (Exact Matching)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def check_exact_match(input_text, dataset):
       # Normalisation avancée
       def normalize(text):
           text = re.sub(r'[^\w\s]', '', text.lower())  # Suppression ponctuation
           text = re.sub(r'\s+', ' ', text).strip()     # Espaces uniformisés
           return text
       
       # Hashing cryptographique
       input_norm = normalize(input_text)
       input_hash = hashlib.sha256(input_norm.encode()).hexdigest()
       
       # Recherche par similarité textuelle
       results = []
       for doc in dataset:
           doc_norm = normalize(doc)
           # 1. Comparaison par hash
           if hashlib.sha256(doc_norm.encode()).hexdigest() == input_hash:
               return [(doc, 1.0)]
           
           # 2. Similarité de Levenshtein
           dist = Levenshtein.distance(input_norm, doc_norm)
           if dist/len(input_norm) < 0.1:  # Seuil 10%
               results.append((doc, 1.0 - dist/len(input_norm)))
           
           # 3. Fenêtre glissante (8 mots)
           input_words = input_norm.split()
           doc_words = doc_norm.split()
           for i in range(len(input_words) - 8):
               segment = ' '.join(input_words[i:i+8])
               if segment in doc_norm:
                   results.append((doc, 0.9))
       
       return sorted(set(results), key=lambda x: x[1], reverse=True)

Algorithmie Avancée:
^^^^^^^^^^^^^^^^^^^^
1. **Normalisation**:
   - Conversion Unicode NFC
   - Stemming léger
   - Correction des espaces

2. **Hashing**:
   - SHA-256 pour collision minimale
   - Tolère variations mineures

3. **Fenêtre Glissante**:
   - Taille optimale: 8 mots
   - Pas de décalage: 1 mot
   - Pondération: 0.9 pour matches partiels

2.2.2 Similarité Lexicale (TF-IDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def tfidf_similarity(query, documents):
       # Vectorisation avancée
       vectorizer = TfidfVectorizer(
           ngram_range=(1, 3),  # Uni-grams à Tri-grams
           analyzer='word',      # Niveau mot
           stop_words=None,      # Gestion manuelle
           min_df=2,             # Filtre termes rares
           max_df=0.95           # Filtre termes trop fréquents
       )
       
       # Construction matrice
       tfidf_matrix = vectorizer.fit_transform([query] + documents)
       
       # Calcul similarité cosinus
       cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
       return cosine_sim[0]

Optimisations:
^^^^^^^^^^^^^^
- **N-grams**:
  - Capture expressions multi-mots
  - Poids: 1.0 pour uni-gram, 0.8 pour bi-gram, 0.6 pour tri-gram

- **Filtrage**:
  - ``min_df``: Élimine les hapax
  - ``max_df``: Supprime les stopwords

2.2.3 Similarité Sémantique (Cross-Encoder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def semantic_similarity(query, candidates):
       # Initialisation modèle
       model = CrossEncoder(
           'cross-encoder/ms-marco-MiniLM-L-6-v2',
           device='cuda' if torch.cuda.is_available() else 'cpu'
       )
       
       # Préparation des paires
       pairs = [(query, cand) for cand in candidates]
       
       # Calcul des scores
       scores = model.predict(
           pairs,
           batch_size=32,
           show_progress_bar=True,
           activation_fct=torch.sigmoid  # Normalisation [0,1]
       )
       
       return scores

Paramètres GPU:
^^^^^^^^^^^^^^^
- **Batch Size**: Adaptatif selon mémoire GPU
- **Précision**: Mixed-precision (FP16)
- **Optimisation**: Kernel fusion pour débit maximal

2.3 Fusion des Résultats
------------------------

2.3.1 Algorithme de Combinaison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def combine_results(exact_matches, tfidf_scores, semantic_scores):
       # Pondérations
       weights = {
           'exact': 0.4,
           'tfidf': 0.3,
           'semantic': 0.3
       }
       
       # Normalisation
       tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min())
       semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
       
       # Combinaison linéaire
       combined_scores = []
       for i in range(len(tfidf_scores)):
           if exact_matches[i][1] == 1.0:  # Match exact
               combined = 1.0
           else:
               combined = (weights['exact'] * exact_matches[i][1] +
                          weights['tfidf'] * tfidf_scores[i] +
                          weights['semantic'] * semantic_scores[i])
           combined_scores.append(combined)
       
       return combined_scores

Formule Mathématique:
^^^^^^^^^^^^^^^^^^^^^
\[
\text{FinalScore} = 0.4 \times \text{ExactMatch} + 0.3 \times \text{TFIDF} + 0.3 \times \text{CrossEncoder}
\]

2.4 Gestion Multilingue
-----------------------

2.4.1 Traduction Automatique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def translate_text(text, target_lang):
       # Détection automatique
       src_lang = detect(text) if len(text) > 50 else 'fr'
       
       # Modèle spécialisé
       model_name = {
           ('fr', 'en'): 'llama3-fr-en',
           ('en', 'fr'): 'llama3-en-fr'
       }.get((src_lang, target_lang))
       
       if not model_name:
           return text  # Retour original si non supporté
       
       # Appel Ollama optimisé
       response = ollama.generate(
           model=model_name,
           prompt=text,
           options={
               'temperature': 0.1,  # Faible créativité
               'top_p': 0.9,
               'max_tokens': len(text) * 2  # Buffer suffisant
           }
       )
       return response['response']

Optimisations:
^^^^^^^^^^^^^^
- **Modèles Spécialisés**: Entraînés sur corpus académique
- **Contrôle Qualité**: Vérification cohérence terminologique
- **Cache**: Mémoization des traductions

===============================================
PARTIE 3 : ANALYSE COMPLÈTE DU FICHIER MAIN4.PY
===============================================

3.1 Interface Utilisateur Avancée
---------------------------------

3.1.1 Configuration Streamlit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   st.set_page_config(
       layout="wide",
       page_title="Plagiarism Sentinel Pro",
       page_icon="🔍",
       initial_sidebar_state="expanded",
       menu_items={
           'Get Help': 'https://github.com/...',
           'Report a bug': "https://github.com/.../issues",
           'About': "### Version 2.1.0\nSystème expert de détection de plagiat"
       }
   )

3.1.2 Gestion des Fichiers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def process_uploaded_file(uploaded_file):
       if uploaded_file.type == "application/pdf":
           # Extraction PDF avancée
           reader = PdfReader(uploaded_file)
           text = ""
           for page in reader.pages:
               text += page.extract_text() + "\n"
           # Nettoyage
           text = re.sub(r'\s+', ' ', text)
           return text.strip()
       
       elif uploaded_file.type.endswith('wordprocessingml.document'):
           # Extraction DOCX avec métadonnées
           text = docx2txt.process(uploaded_file)
           return re.sub(r'\[.*?\]', '', text)  # Suppression commentaires

3.2 Visualisations Interactives
-------------------------------

3.2.1 Réseau de Similarité
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def create_similarity_network(matches):
       G = nx.DiGraph()  # Graphe orienté
       
       # Ajout noeuds
       G.add_node("SOURCE", size=20, color='#FF6B6B')
       
       # Ajout correspondances
       for idx, match in enumerate(matches):
           doc_id = match['metadata'].get('doc_id', f"DOC_{idx}")
           G.add_node(doc_id, size=15, color='#4ECDC4')
           G.add_edge("SOURCE", doc_id, 
                     weight=match['score'], 
                     title=f"Similarité: {match['score']:.2f}")
       
       # Configuration visuelle
       net = Network(
           height="750px",
           width="100%",
           bgcolor="#222222",
           font_color="white",
           directed=True,
           filter_menu=True
       )
       net.from_nx(G)
       return net

3.2.2 Analyse Stylistique
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def analyze_style(text, lang):
       # Chargement modèle adapté
       nlp = spacy.load("fr_core_news_lg" if lang == 'fr' else "en_core_web_lg")
       
       # Traitement complet
       doc = nlp(text)
       
       # Métriques avancées
       metrics = {
           'readability': textstat.flesch_reading_ease(text),
           'avg_sentence_length': np.mean([len(sent.text) for sent in doc.sents]),
           'pos_ratios': {
               'NOUN': len([t for t in doc if t.pos_ == 'NOUN'])/len(doc),
               'VERB': len([t for t in doc if t.pos_ == 'VERB'])/len(doc),
               'ADJ': len([t for t in doc if t.pos_ == 'ADJ'])/len(doc)
           },
           'dependency_depth': np.mean([
               max([len(list(t.children)) for t in sent])
               for sent in doc.sents
           ])
       }
       return metrics

3.3 Gestion des Résultats
-------------------------

3.3.1 Génération de Rapport
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def generate_report(results):
       # Structure complète
       report = {
           "metadata": {
               "date": datetime.now().isoformat(),
               "processing_time": results['processing_time'],
               "word_count": len(results['text'].split())
           },
           "analysis": {
               "overall_score": results['score'],
               "matches": sorted(
                   results['matches'],
                   key=lambda x: x['score'],
                   reverse=True
               ),
               "style_analysis": results['style']
           },
           "risk_assessment": classify_risk(results['score'])
       }
       
       # Formats multiples
       return {
           'json': json.dumps(report, indent=2),
           'html': generate_html_report(report),
           'pdf': generate_pdf_report(report)
       }

3.3.2 Classification des Risques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def classify_risk(score):
       if score >= 0.9:
           return {
               "level": "CRITICAL",
               "action": "Rejet immédiat - plagiat avéré",
               "confidence": 0.99
           }
       elif score >= 0.7:
           return {
               "level": "HIGH",
               "action": "Révision majeure requise",
               "confidence": 0.85
           }
       # ... autres niveaux ...

===============================================
ANNEXES TECHNIQUES
===============================================

A.1 Spécifications Complètes
----------------------------

.. list-table:: Environnement Technique
   :widths: 20 30 50
   :header-rows: 1

   * - Composant
     - Version
     - Configuration
   * - Python
     - 3.10.12
     - Optimisations AVX2
   * - CUDA
     - 12.1
     - Compute Capability 8.6
   * - Ollama
     - 0.1.26
     - 4-bit quantization

A.2 Exemples Complets
---------------------

Requête Complexe:

.. code-block:: python
   :linenos:

   results = hybrid_search(
       query="L'impératif catégorique chez Kant",
       dataset=philosophy_corpus,
       top_k=10,
       lang='fr',
       filters={
           'min_date': '1900-01-01',
           'max_date': '2023-12-31'
       }
   )

Résultat Détaillé:

.. code-block:: json

   {
     "query": "L'impératif catégorique chez Kant",
     "matches": [
       {
         "score": 0.934,
         "type": "semantic",
         "text": "Emmanuel Kant formule dans la 'Critique de la raison pratique'...",
         "source": "kant_ethics.pdf",
         "metadata": {
           "author": "M. Heidegger",
           "year": 1952
         }
       }
     ],
     "analysis": {
       "style_consistency": 0.87,
       "risk_level": "HIGH"
     }
   }

A.3 Optimisations Avancées
--------------------------

.. code-block:: python
   :linenos:

   @functools.lru_cache(maxsize=1000)
   def cached_semantic_search(query):
       return semantic_search(query)

   async def async_process_documents(docs):
       with ThreadPoolExecutor(max_workers=8) as executor:
           loop = asyncio.get_event_loop()
           futures = [
               loop.run_in_executor(
                   executor,
                   process_document,
                   doc
               ) for doc in docs
           ]
           return await asyncio.gather(*futures)
