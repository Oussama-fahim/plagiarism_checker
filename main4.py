import ollama
import streamlit as st
import numpy as np
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langdetect import detect, DetectorFactory
from typing import List, Dict, Any, Tuple
import time
import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import plotly.express as px
from collections import defaultdict
import re
import hashlib
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import json
import networkx as nx
from pyvis.network import Network
import tempfile
import spacy
from wordcloud import WordCloud
import textstat
from streamlit.components.v1 import html
import docx2txt
import PyPDF2
import base64
from annotated_text import annotated_text
from st_aggrid import AgGrid, GridOptionsBuilder

# Initialisation des modèles NLP
try:
    nlp_en = spacy.load("en_core_web_lg")
    nlp_fr = spacy.load("fr_core_news_sm")
except:
    st.warning("Modèles SpaCy non trouvés. Installation requise : python -m spacy download en_core_web_lg && python -m spacy download fr_core_news_lg")
    st.stop()

# Configurations initiales
DetectorFactory.seed = 0
st.set_page_config(layout="wide", page_title="🔍 AI Plagiarism Sentinel Pro", page_icon="🔍")

# Initialisation des variables globales
vecdb = None
tfidf_vectorizer = None

# Initialisation du cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==============================================
# FONCTIONS UTILITAIRES AVANCÉES
# ==============================================

def load_assets():
    """Charge les assets visuels et retourne le banner"""
    try:
        response = requests.get("https://unsplash.com/fr/photos/un-ecran-dordinateur-avec-des-mots-faux-et-faux-dessus-0qCgDhRADxY")
        banner = Image.open(BytesIO(response.content))
        return banner
    except:
        return None

def get_base64_image(image_path):
    """Convertit une image en base64 pour le CSS"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def apply_custom_css():
    """Applique le CSS personnalisé"""
    css = f"""
    <style>
        .main {{
            background-color: #f9f9f9;
        }}
        .header {{
            background: linear-gradient(135deg, #434343 0%, #000000 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .exact-match {{ 
            border-left: 6px solid #ef4444; 
            background-color: rgba(239, 68, 68, 0.05); 
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        .partial-match {{ 
            border-left: 6px solid #f59e0b; 
            background-color: rgba(245, 158, 11, 0.05);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        .semantic-match {{ 
            border-left: 6px solid #10b981; 
            background-color: rgba(16, 185, 129, 0.05);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        .sentence-match {{ 
            background-color: rgba(255, 237, 213, 0.7); 
            padding: 2px 6px; 
            border-radius: 4px; 
        }}
        .idea-match {{ 
            background-color: rgba(173, 216, 230, 0.3); 
            padding: 10px; 
            border-radius: 8px; 
            margin: 5px 0; 
        }}
        .result-card {{
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .progress-bar {{
            height: 10px;
            border-radius: 5px;
            margin-top: 0.5rem;
        }}
        .tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, #1e3a8a, #1e40af);
        }}
        .stButton>button {{
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: white;
            border: none;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==============================================
# FONCTIONS CORE DE DÉTECTION
# ==============================================

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialise les modèles et bases de données"""
    global vecdb, tfidf_vectorizer
    
    try:
        with st.spinner('🚀 Initialisation des modèles IA...'):
            progress_bar = st.progress(0)
            
            # Phase 1: Embeddings
            progress_bar.progress(20)
            embeddings = OllamaEmbeddings(
                model="mxbai-embed-large:latest",
                temperature=0.01,
                top_k=50
            )
            
            # Phase 2: Base vectorielle
            progress_bar.progress(50)
            vecdb = Chroma(
                persist_directory="philo_db",
                embedding_function=embeddings,
                collection_name="rag-chroma"
            )
            
            # Phase 3: Dataset
            progress_bar.progress(70)
            dataset_contents = []
            if hasattr(vecdb, '_collection'):
                dataset_contents = vecdb._collection.get(include=['documents'])['documents']
            
            # Phase 4: Vectorizer
            progress_bar.progress(90)
            tfidf_vectorizer = TfidfVectorizer(
                stop_words=None,
                ngram_range=(1, 3),
                analyzer='word'
            )
            if dataset_contents:
                tfidf_vectorizer.fit(dataset_contents)
            
            progress_bar.progress(100)
            time.sleep(0.5)
            
            return vecdb, embeddings, dataset_contents, tfidf_vectorizer
    
    except Exception as e:
        st.error(f"🚨 Erreur critique: {str(e)}")
        st.stop()

def check_exact_match(input_text: str, dataset: List[str]) -> List[Tuple[str, float]]:
    """Vérifie les correspondances exactes avec normalisation avancée"""
    def normalize(text):
        text = re.sub(r'[^\w\s]', '', text.strip().lower())
        return re.sub(r'\s+', ' ', text)
    
    normalized_input = normalize(input_text)
    input_hash = hashlib.md5(normalized_input.encode('utf-8')).hexdigest()
    matches = []
    
    for doc in dataset:
        normalized_doc = normalize(doc)
        doc_hash = hashlib.md5(normalized_doc.encode('utf-8')).hexdigest()
        
        if input_hash == doc_hash:
            return [(doc, 1.0)]
        
        # Similarité textuelle indépendante de la langue
        match_ratio = SequenceMatcher(None, normalized_input, normalized_doc).ratio()
        if match_ratio > 0.7:
            matches.append((doc, match_ratio))
        
        # Vérification des segments longs
        input_words = normalized_input.split()
        doc_words = normalized_doc.split()
        
        for i in range(len(input_words) - 8 + 1):  # Fenêtre de 8 mots
            segment = ' '.join(input_words[i:i+8])
            if segment in normalized_doc:
                matches.append((doc, max(match_ratio, 0.85)))
                break
    
    unique_matches = {match[0]: match[1] for match in matches}
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

@st.cache_data(ttl=3600, show_spinner=False)
def translate_text(text: str, target_lang: str) -> str:
    """Traduction intelligente avec gestion des erreurs"""
    try:
        if len(text) < 50:  # Ne pas traduire les textes trop courts
            return text
            
        response = ollama.chat(
            model="llama3.1",
            messages=[{
                "role": "system",
                "content": f"Traduis ce texte en {target_lang} en conservant le sens original:\n{text}"
            }],
            options={'temperature': 0.1}
        )
        return response["message"]["content"]
    except Exception as e:
        st.warning(f"Traduction partielle: {str(e)}")
        return text

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcule la similarité combinée TF-IDF + Cross-Encoder"""
    global tfidf_vectorizer
    
    try:
        # Similarité lexicale (TF-IDF)
        vectors = tfidf_vectorizer.transform([text1, text2])
        tfidf_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Similarité sémantique (Cross-Encoder)
        cross_score = cross_encoder.predict([[text1, text2]])[0]
        
        # Combinaison pondérée
        return (cross_score * 0.7) + (tfidf_sim * 0.3)
    except Exception as e:
        st.warning(f"Calcul de similarité simplifié: {str(e)}")
        return SequenceMatcher(None, text1, text2).ratio()

def hybrid_search(query: str, dataset: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    """Recherche hybride multilingue avec gestion des erreurs"""
    global vecdb
    
    try:
        # Détection de la langue de la requête
        query_lang = detect(query) if len(query) > 20 else 'en'
        
        # 1. Vérifier les copies exactes
        exact_matches = check_exact_match(query, dataset)
        if exact_matches:
            return [{
                "content": match[0],
                "similarity": match[1],
                "match_type": "exact",
                "metadata": {},
                "combined_score": match[1]
            } for match in exact_matches[:top_k]]
        
        # 2. Recherche dans la langue d'origine
        vector_results = vecdb.similarity_search_with_score(query, k=top_k*2)
        
        # 3. Si la requête est en français, chercher aussi en anglais et vice versa
        translated_results = []
        if query_lang == 'fr':
            translated_query = translate_text(query, 'en')
            if translated_query != query:
                translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k)
        elif query_lang == 'en':
            translated_query = translate_text(query, 'fr')
            if translated_query != query:
                translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k)
        
        # Combiner les résultats
        all_results = []
        
        # Ajouter les résultats originaux
        for doc, score in vector_results:
            sim_score = calculate_similarity(query, doc.page_content)
            all_results.append({
                "content": doc.page_content,
                "similarity": sim_score,
                "match_type": "semantic",
                "metadata": doc.metadata,
                "combined_score": sim_score
            })
        
        # Ajouter les résultats traduits
        for doc, score in translated_results:
            translated_content = translate_text(doc.page_content, query_lang)
            sim_score = calculate_similarity(query, translated_content)
            all_results.append({
                "content": doc.page_content,
                "similarity": sim_score,
                "match_type": "translated",
                "metadata": doc.metadata,
                "combined_score": sim_score * 0.9  # Légère pénalité pour la traduction
            })
        
        # Éliminer les doublons et trier
        unique_results = {}
        for res in all_results:
            if res["content"] not in unique_results or res["combined_score"] > unique_results[res["content"]]["combined_score"]:
                unique_results[res["content"]] = res
        
        return sorted(unique_results.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]
    
    except Exception as e:
        st.error(f"Erreur de recherche: {str(e)}")
        return []

def analyze_ideas(input_text: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyse des similarités conceptuelles entre phrases"""
    ideas = []
    sentences = [s.strip() for s in re.split(r'[.!?]', input_text) if len(s.strip().split()) > 5]
    
    for match in matches:
        if match["combined_score"] < 0.4:  # Seuil pour les idées similaires
            continue
            
        match_sentences = [s.strip() for s in re.split(r'[.!?]', match["content"]) if len(s.strip().split()) > 5]
        
        for sent in sentences:
            for match_sent in match_sentences:
                sim_score = calculate_similarity(sent, match_sent)
                if sim_score > 0.5:  # Seuil pour similarité d'idée
                    ideas.append({
                        "source_sentence": sent,
                        "matched_sentence": match_sent,
                        "similarity": sim_score,
                        "source_content": match["content"][:200] + "...",
                        "metadata": match.get("metadata", {})
                    })
    
    # Regrouper les idées similaires
    grouped_ideas = defaultdict(list)
    for idea in ideas:
        key = idea["source_sentence"][:50]  # Regrouper par phrase source
        grouped_ideas[key].append(idea)
    
    # Garder la meilleure correspondance pour chaque groupe
    return [max(group, key=lambda x: x["similarity"]) for group in grouped_ideas.values()]

def analyze_writing_style(text: str, language: str) -> Dict[str, Any]:
    """Analyse approfondie du style d'écriture"""
    doc = nlp_en(text) if language == 'en' else nlp_fr(text)
    
    metrics = {
        "readability": textstat.flesch_reading_ease(text),
        "complexity": textstat.dale_chall_readability_score(text),
        "sentiment": sum([sent.sentiment for sent in doc.sents]) / len(list(doc.sents)),
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "avg_word_length": np.mean([len(token.text) for token in doc if not token.is_punct]),
        "pos_tags": {tag: sum(1 for token in doc if token.pos_ == tag) for tag in set([token.pos_ for token in doc])},
        "lexical_diversity": len(set([token.text.lower() for token in doc if not token.is_punct])) / 
                            len([token.text for token in doc if not token.is_punct]),
        "avg_sentence_length": np.mean([len(sent) for sent in doc.sents])
    }
    
    return metrics

def create_similarity_network(matches: List[Dict[str, Any]]) -> str:
    """Crée un réseau de similarité interactif"""
    G = nx.Graph()
    
    for i, match in enumerate(matches):
        G.add_node(f"Source_{i}", size=15, color='blue')
        G.add_node(match['metadata'].get('source', f"Doc_{i}"), size=10, color='red')
        G.add_edge(f"Source_{i}", match['metadata'].get('source', f"Doc_{i}"), 
                  weight=match['combined_score'], title=f"{match['combined_score']:.2f}")
    
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # Sauvegarde temporaire pour affichage
    path = tempfile.mkdtemp()
    net.save_graph(f"{path}/network.html")
    
    return open(f"{path}/network.html").read()

def generate_wordcloud(matches: List[Dict[str, Any]]):
    """Génère un nuage de mots à partir des correspondances"""
    text = " ".join([match['content'] for match in matches])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def display_style_analysis(style_metrics: Dict[str, Any]):
    """Affiche l'analyse stylistique sous forme de dashboard"""
    st.markdown("### 📊 Métriques de lisibilité")
    col1, col2, col3 = st.columns(3)
    col1.metric("Score de lisibilité", f"{style_metrics['readability']:.1f}", 
                "Facile" if style_metrics['readability'] > 60 else "Difficile")
    col2.metric("Complexité", f"{style_metrics['complexity']:.1f}")
    col3.metric("Diversité lexicale", f"{style_metrics['lexical_diversity']*100:.1f}%")
    
    st.markdown("### 📝 Analyse grammaticale")
    st.write("**Répartition des parties du discours:**")
    pos_df = pd.DataFrame.from_dict(style_metrics['pos_tags'], orient='index', columns=['Count'])
    st.bar_chart(pos_df)
    
    st.markdown("### 🏷️ Entités nommées détectées")
    if style_metrics['entities']:
        entities_df = pd.DataFrame(style_metrics['entities'], columns=['Entité', 'Type'])
        st.dataframe(entities_df.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.info("Aucune entité nommée détectée")

def generate_full_report(results: Dict[str, Any], style_analysis: Dict[str, Any]) -> str:
    """Génère un rapport complet au format JSON"""
    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_duration": results.get('processing_time', 0),
            "total_matches": len(results.get('all_matches', []))
        },
        "results": {
            "highest_score": results.get('highest_score', 0),
            "average_score": results.get('average_score', 0),
            "exact_matches": results.get('exact_matches', []),
            "partial_matches": results.get('partial_matches', []),
            "semantic_matches": results.get('semantic_matches', []),
            "similar_ideas": results.get('similar_ideas', [])
        },
        "style_analysis": style_analysis
    }
    return json.dumps(report, indent=2, ensure_ascii=False)

def process_results(matches: List[Dict[str, Any]], input_text: str) -> Dict[str, Any]:
    """Traite et organise les résultats pour l'affichage"""
    analysis = {
        "exact_matches": [m for m in matches if m["match_type"] == "exact"],
        "partial_matches": [m for m in matches if m["combined_score"] > 0.7 and m["match_type"] != "exact"],
        "semantic_matches": [m for m in matches if m["combined_score"] <= 0.7],
        "highest_score": max([m["combined_score"] for m in matches]) * 100 if matches else 0,
        "average_score": (sum([m["combined_score"] for m in matches]) / len(matches)) * 100 if matches else 0,
        "all_matches": matches,
        "similar_ideas": analyze_ideas(input_text, matches)
    }
    return analysis

def display_match_card(match: Dict[str, Any]):
    """Affiche une carte visuelle pour chaque correspondance"""
    match_type = {
        "exact": "Copie exacte",
        "semantic": "Similarité sémantique",
        "translated": "Correspondance multilingue"
    }.get(match["match_type"], "Autre")
    
    with st.container():
        st.markdown(f"""
        <div class="{'exact-match' if match['match_type'] == 'exact' else 'partial-match' if match['combined_score'] > 0.7 else 'semantic-match'}">
            <h3>{match_type} - Score: {match['combined_score']*100:.1f}%</h3>
            <p><strong>Source:</strong> {match['metadata'].get('source', 'Inconnue')}</p>
            <div class="progress-bar">
                <div style="width: {match['combined_score']*100}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Extrait analysé**")
            st.text(match.get('source_excerpt', match['content'][:200] + '...'))
        with col2:
            st.markdown("**Texte correspondant**")
            st.text(match['content'][:200] + ('...' if len(match['content']) > 200 else ''))

# ==============================================
# INTERFACE UTILISATEUR PRINCIPALE
# ==============================================

def main():
    # Initialisation
    global vecdb, tfidf_vectorizer
    vecdb, embeddings, dataset_contents, tfidf_vectorizer = initialize_system()
    banner_image = load_assets()
    apply_custom_css()
    
    # En-tête personnalisé
    if banner_image:
        st.image(banner_image, use_column_width=True, caption="AI Plagiarism Sentinel Pro - Détection avancée de similarités textuelles")
    else:
        st.markdown("""
        <div class="header">
            <h1 style="color: white; margin: 0;">🔍 AI Plagiarism Sentinel Pro</h1>
            <p style="color: #aaa; margin: 0;">Détection intelligente multi-niveaux avec analyse stylistique avancée</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar amélioré
    with st.sidebar:
        st.title("⚙️ Paramètres Experts")
        
        with st.expander("🔍 Options de Recherche", expanded=True):
            analysis_mode = st.selectbox(
                "Mode d'analyse",
                ["DeepScan Pro", "Rapide", "Manuel Expert"],
                help="DeepScan Pro: Analyse approfondie mais plus lente"
            )
            
            sensitivity = st.slider(
                "Niveau de sensibilité",
                1, 10, 8,
                help="Contrôle l'agressivité de la détection"
            )
            
            similarity_threshold = st.slider(
                "Seuil de similarité minimum",
                0.1, 1.0, 0.5,
                help="Filtre les résultats en dessous de ce seuil"
            )
        
        with st.expander("📊 Visualisation", expanded=False):
            viz_options = st.multiselect(
                "Options de visualisation",
                ["Réseau de similarité", "Nuage de mots", "Analyse stylistique"],
                default=["Réseau de similarité"]
            )
        
        st.divider()
        
        # Statistiques en temps réel
        st.markdown("### 📈 Statistiques Live")
        col1, col2 = st.columns(2)
        col1.metric("Documents indexés", vecdb._collection.count() if hasattr(vecdb, '_collection') else "N/A")
        col2.metric("Taille du corpus", f"{len(dataset_contents)} textes")
        
        # Bouton d'export
        st.download_button(
            label="📤 Exporter le rapport",
            data="",  # Serait rempli après analyse
            file_name="plagiarism_report.json",
            mime="application/json",
            disabled=True,
            key="initial_download"
        )
    
    # Zone de saisie améliorée
    input_text = ""
    with st.container():
        input_method = st.radio(
            "Source d'entrée",
            ["📝 Texte direct", "📂 Fichier", "🌐 URL"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_method == "📝 Texte direct":
            input_text = st.text_area(
                "Saisissez votre texte à analyser:",
                height=300,
                placeholder="Collez ici le contenu à vérifier...",
                label_visibility="collapsed",
                key="text_input"
            )
        elif input_method == "📂 Fichier":
            uploaded_file = st.file_uploader(
                "Téléversez un document",
                type=["txt", "pdf", "docx", "pptx", "odt"],
                label_visibility="collapsed",
                key="file_uploader"
            )
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    input_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    input_text = docx2txt.process(uploaded_file)
                else:
                    input_text = uploaded_file.getvalue().decode("utf-8")
        else:
            url = st.text_input("Entrez une URL à analyser", placeholder="https://example.com/article.html", key="url_input")
            if url:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers)
                    input_text = response.text
                except Exception as e:
                    st.error(f"Impossible de récupérer le contenu de l'URL: {str(e)}")
    
    # Bouton d'analyse avec effet visuel
    if st.button("🚀 Lancer l'analyse approfondie", use_container_width=True, key="analyze_button"):
        if not input_text or len(input_text.split()) < 20:
            st.warning("Le texte est trop court (minimum 20 mots requis)")
        else:
            with st.spinner("🔍 Analyse en cours avec nos algorithmes IA..."):
                start_time = time.time()
                
                # Détection de langue avancée
                try:
                    lang = detect(input_text[:1000]) if len(input_text) > 50 else 'fr'
                    st.info(f"Langue détectée: {'Français' if lang == 'fr' else 'Anglais'}")
                except:
                    lang = 'fr'
                    st.warning("Détection de langue échouée, utilisation du français par défaut")
                
                # Analyse de style
                style_analysis = analyze_writing_style(input_text, lang)
                
                # Recherche de similarité
                matches = hybrid_search(input_text, dataset_contents, top_k=20)
                
                # Traitement des résultats
                results = process_results(matches, input_text)
                results['processing_time'] = time.time() - start_time
                
                # Génération du rapport
                full_report = generate_full_report(results, style_analysis)
                
                # Mise à jour du bouton de téléchargement
                st.session_state.download_report = full_report
                st.session_state.download_disabled = False
                
                # Affichage des résultats
                display_results(results, style_analysis, viz_options, results['processing_time'], input_text)

def display_results(results: Dict[str, Any], style_analysis: Dict[str, Any], viz_options: List[str], processing_time: float, input_text: str):
    """Affiche les résultats de l'analyse"""
    st.success(f"✅ Analyse terminée en {processing_time:.2f} secondes")
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Score maximal", f"{results['highest_score']:.1f}%")
    col2.metric("Correspondances", len(results['all_matches']))
    col3.metric("Originalité", f"{100 - results['highest_score']:.1f}%")
    
    # Score global avec indicateur visuel
    score_global = results["highest_score"]
    if score_global > 80:
        status = "PLAGIAT ÉVIDENT"
        status_class = "exact-match"
        status_emoji = "🔴"
    elif score_global > 60:
        status = "PLAGIAT PROBABLE"
        status_class = "partial-match"
        status_emoji = "🟠"
    elif score_global > 40:
        status = "SIMILITUDES"
        status_class = "semantic-match"
        status_emoji = "🟡"
    else:
        status = "PAS DE PLAGIAT"
        status_class = ""
        status_emoji = "🟢"
    
    st.markdown(f"""
    <div class="result-card {status_class}">
        <h2>{status_emoji} {status} - Score: {score_global:.1f}%</h2>
        <p>Temps d'analyse: {processing_time:.2f}s | Score moyen: {results['average_score']:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Onglets avancés
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔍 Correspondances", 
        "📈 Visualisations", 
        "✍️ Style", 
        "🛡️ Rapport complet"
    ])
    
    with tab1:
        # Graphique Plotly interactif
        fig = px.pie(
            names=["Original", "Similitudes"],
            values=[100 - results['highest_score'], results['highest_score']],
            hole=0.5,
            color_discrete_sequence=['green', 'red']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Aperçu des correspondances
        st.markdown("### 🔝 Principales correspondances")
        for match in results['all_matches'][:3]:
            display_match_card(match)
        
        # Analyse des idées similaires
        if results['similar_ideas']:
            st.markdown("### 💡 Idées conceptuellement similaires")
            for idea in sorted(results['similar_ideas'], key=lambda x: x['similarity'], reverse=True)[:5]:
                st.markdown(f"""
                <div class="idea-match">
                    <b>Similarité: {idea['similarity']*100:.1f}%</b>
                    <p><u>Votre texte:</u> {idea['source_sentence']}</p>
                    <p><u>Texte similaire:</u> {idea['matched_sentence']}</p>
                    <small>Source: {idea['source_content']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # Affichage enrichi des correspondances
        st.markdown(f"### 📋 Liste complète des correspondances ({len(results['all_matches'])})")
        
        # Filtrage interactif
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider(
                "Score minimum à afficher",
                0.0, 1.0, 0.5,
                key="min_score_filter"
            )
        with col2:
            match_types = st.multiselect(
                "Types de correspondances",
                ["exact", "semantic", "translated"],
                default=["exact", "semantic", "translated"],
                key="match_type_filter"
            )
        
        filtered_matches = [
            m for m in results['all_matches'] 
            if m['combined_score'] >= min_score and m['match_type'] in match_types
        ]
        
        if not filtered_matches:
            st.info("Aucune correspondance ne correspond aux critères de filtrage")
        else:
            for match in filtered_matches:
                with st.expander(f"{match['match_type'].capitalize()} - Score: {match['combined_score']*100:.1f}%", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Votre texte**")
                        st.markdown(f"```\n{match.get('source_excerpt', match['content'][:200])}\n```")
                    with col2:
                        st.markdown("**Texte correspondant**")
                        st.markdown(f"```\n{match['content']}\n```")
                    
                    st.markdown(f"**Source:** `{match['metadata'].get('source', 'Inconnue')}`")
                    st.progress(match['combined_score'])
    
    with tab3:
        if "Réseau de similarité" in viz_options:
            st.markdown("### 🕸️ Réseau des similarités")
            html_content = create_similarity_network(results['all_matches'])
            html(html_content, height=600, scrolling=True)
        
        if "Nuage de mots" in viz_options:
            st.markdown("### ☁️ Nuage de mots clés")
            generate_wordcloud(results['all_matches'])
        
        # Graphique supplémentaire
        st.markdown("### 📊 Répartition des types de correspondances")
        match_types = {
            "Copies exactes": len(results['exact_matches']),
            "Paraphrases": len(results['partial_matches']),
            "Similarités": len(results['semantic_matches'])
        }
        fig, ax = plt.subplots()
        ax.bar(match_types.keys(), match_types.values(), color=['red', 'orange', 'blue'])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    
    with tab4:
        st.markdown("### ✨ Analyse stylistique approfondie")
        display_style_analysis(style_analysis)
        
        # Visualisation POS tags
        st.markdown("### 📌 Répartition des POS tags")
        pos_df = pd.DataFrame.from_dict(style_analysis['pos_tags'], orient='index', columns=['Count'])
        st.bar_chart(pos_df)
        
        # Nuage de mots personnalisé
        st.markdown("### ☁️ Votre profil lexical")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with tab5:
        st.markdown("## 📑 Rapport complet d'analyse")
        
        # Affichage du rapport JSON
        with st.expander("📝 Voir le rapport technique complet", expanded=False):
            st.code(generate_full_report(results, style_analysis), language='json')
        
        # Bouton de téléchargement dynamique
        st.download_button(
            label="📥 Télécharger le rapport complet",
            data=st.session_state.get('download_report', ''),
            file_name=f"plagiarism_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            disabled=st.session_state.get('download_disabled', True)
        )
        
        # Résumé exécutif
        st.markdown("### 📌 Résumé exécutif")
        st.markdown(f"""
        - **Score de plagiat maximal**: {results['highest_score']:.1f}%
        - **Nombre total de correspondances**: {len(results['all_matches'])}
        - **Copies exactes détectées**: {len(results['exact_matches'])}
        - **Paraphrases détectées**: {len(results['partial_matches'])}
        - **Idées similaires identifiées**: {len(results['similar_ideas'])}
        - **Niveau de lisibilité**: {'Élevé' if style_analysis['readability'] > 60 else 'Moyen' if style_analysis['readability'] > 30 else 'Faible'}
        - **Diversité lexicale**: {style_analysis['lexical_diversity']*100:.1f}%
        """)
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        if results['highest_score'] > 80:
            st.error("**Action requise**: Ce texte présente des signes évidents de plagiat. Il est recommandé de le réécrire complètement ou de citer correctement les sources.")
        elif results['highest_score'] > 60:
            st.warning("**Attention nécessaire**: Ce texte contient des passages probablement plagiés. Vérifiez les sources et ajoutez des citations appropriées.")
        elif results['highest_score'] > 40:
            st.info("**Vérification suggérée**: Certaines similarités ont été détectées. Assurez-vous que toutes les sources sont correctement attribuées.")
        else:
            st.success("**Texte original**: Aucun signe de plagiat détecté. Le texte semble majoritairement original.")

if __name__ == "__main__":
    main()