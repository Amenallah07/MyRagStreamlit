"""
Interface Streamlit pour le Chatbot RAG - Version Cloud
========================================================

Interface web pour les recruteurs (compatible Elastic Cloud + Streamlit Cloud)
"""

import streamlit as st
import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from groq import Groq
import time
from datetime import datetime
import re

# ============================================================================
# CONFIGURATION - Version Cloud
# ============================================================================

# Configuration Elastic Cloud
# Utilise st.secrets pour Streamlit Cloud, sinon variables d'environnement
try:
    ES_CLOUD_URL = st.secrets["ES_CLOUD_URL"]  # Streamlit Cloud
except:
    ES_CLOUD_URL = os.getenv("ES_CLOUD_URL")   # Railway, Docker, VPS

try:
    ES_API_KEY = st.secrets["ES_API_KEY"]  # Streamlit Cloud
except:
    ES_API_KEY = os.getenv("ES_API_KEY")   # Railway, Docker, VPS

INDEX_NAME = "cv_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Streamlit Cloud
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # Railway, Docker, VPS


GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_RELEVANCE_SCORE = 5.0

# Configuration Streamlit
st.set_page_config(
    page_title="Assistant RH - Recherche de Candidats",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# INITIALISATION (avec cache)
# ============================================================================

@st.cache_resource
def init_clients():
    """Initialise les clients (cache pour éviter rechargement)"""
    # Connexion Elastic Cloud avec API Key
    es = Elasticsearch(
        [ES_CLOUD_URL],
        api_key=ES_API_KEY,
        request_timeout=30
    )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    groq_client = Groq(api_key=GROQ_API_KEY)

    return es, embedding_model, groq_client


# ============================================================================
# FONCTIONS DE RECHERCHE (identiques)
# ============================================================================

def classify_query_with_llm(query, groq_client):
    """Classification LLM"""
    classification_prompt = f"""Détermine si cette question concerne le recrutement, les ressources humaines ou la recherche de candidats.

Question: "{query}"

Réponds UNIQUEMENT par "OUI" ou "NON".

Exemples de questions RH (OUI):
- "Trouve un Data Scientist"
- "Qui maîtrise Python?"
- "Cherche un développeur React"

Exemples de questions hors RH (NON):
- "Capitale de la France?"
- "Météo à Paris?"

Réponse (OUI ou NON):"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "Tu es un classificateur de questions. Réponds UNIQUEMENT par OUI ou NON."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip().upper()
        return "OUI" in answer
    except:
        return False


def deduplicate_results(results, max_per_candidate=1):
    """Déduplication"""
    seen_candidates = {}
    deduplicated = []

    for hit in results['hits']['hits']:
        candidat = hit['_source']['candidat_nom']
        if candidat not in seen_candidates:
            seen_candidates[candidat] = 0
        if seen_candidates[candidat] < max_per_candidate:
            deduplicated.append(hit)
            seen_candidates[candidat] += 1

    return deduplicated


def search_candidates(query, es, embedding_model, min_experience=0, max_results=10):
    """Recherche hybride"""
    query_embedding = embedding_model.encode(query).tolist()

    result = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["chunk_text^2", "candidat_poste^3", "competences_liste^2"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "filter": [
                        {"range": {"candidat_experience": {"gte": min_experience}}}
                    ] if min_experience > 0 else []
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 20,
                "num_candidates": 100
            },
            "size": 50,
            "_source": ["chunk_id", "candidat_nom", "candidat_poste", "candidat_experience", "chunk_text",
                        "competences_liste", "chunk_type"]
        }
    )

    deduplicated = deduplicate_results(result, max_per_candidate=1)
    filtered = [hit for hit in deduplicated if hit['_score'] >= MIN_RELEVANCE_SCORE]
    return filtered[:max_results]


def build_context(candidates):
    """Construit le contexte pour le LLM"""
    if not candidates:
        return "Aucun candidat trouvé."

    context_parts = []
    for i, hit in enumerate(candidates, 1):
        doc = hit['_source']
        competences = doc.get('competences_liste', [])[:15]

        candidat_info = f"""Candidat {i}: {doc['candidat_nom']}
- Poste: {doc.get('candidat_poste', 'N/A')}
- Expérience: {doc.get('candidat_experience', 'N/A')} ans
- Score: {hit['_score']:.2f}
- Compétences: {', '.join(competences) if competences else 'N/A'}
- Contexte: {doc['chunk_text'][:250]}"""

        context_parts.append(candidat_info.strip())

    return "\n\n".join(context_parts)


def generate_response(query, candidates, groq_client):
    """Génère la réponse avec le LLM"""
    context = build_context(candidates)

    system_prompt = """Tu es un assistant RH intelligent spécialisé dans le recrutement.

Ton rôle:
- Analyser les profils de candidats
- Recommander les meilleurs candidats
- Justifier avec des faits précis
- Être concis et professionnel

Format de réponse:
1. Recommandation principale (1-2 candidats top avec score > 10)
2. Justification courte (compétences + expérience)
3. Alternatives si pertinent
4. Résumé en 1 phrase

Règles:
- Si score < 10, mentionne que le candidat est moins pertinent
- Si TOUS les scores < 5, dis qu'aucun candidat pertinent n'est trouvé
- Base-toi UNIQUEMENT sur les infos fournies
- Ne fabrique JAMAIS d'informations"""

    user_prompt = f"""Question: {query}

Candidats disponibles:
{context}

Analyse et réponds de manière structurée."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1200,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur: {e}"


def extract_min_experience(query):
    """Extrait l'expérience minimum de la requête"""
    if "senior" in query.lower():
        return 7
    elif "junior" in query.lower():
        return 0

    exp_match = re.search(r'(\d+)\+?\s*ans?', query, re.IGNORECASE)
    if exp_match:
        return int(exp_match.group(1))

    return 0


# ============================================================================
# INTERFACE STREAMLIT (identique)
# ============================================================================

def main():
    # Chargement des clients
    try:
        es, embedding_model, groq_client = init_clients()

        # Test de connexion
        if not es.ping():
            st.error("❌ Impossible de se connecter à Elasticsearch Cloud")
            st.stop()

    except Exception as e:
        st.error(f"❌ Erreur d'initialisation: {e}")
        st.info("Vérifiez vos identifiants Elastic Cloud dans les secrets Streamlit")
        st.stop()

    # Header
    st.title("👔 Assistant RH - Recherche de Candidats")
    st.markdown("**Propulsé par RAG + Llama 3.3 70B**")
    st.markdown("*🌐 Version Cloud - Elastic Cloud + Streamlit Cloud*")
    st.divider()

    # Sidebar - Paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")

        # Statistiques base de données
        try:
            stats = es.indices.stats(index=INDEX_NAME)
            doc_count = stats['indices'][INDEX_NAME]['total']['docs']['count']

            st.metric("CVs indexés", f"{doc_count // 3}")  # ~3-4 chunks par CV
            st.metric("Chunks totaux", doc_count)
            st.success("✅ Connecté à Elastic Cloud")
        except Exception as e:
            st.warning(f"⚠️ Impossible de récupérer les stats: {e}")

        st.divider()

        # Filtres
        st.subheader("Filtres de recherche")

        use_auto_experience = st.checkbox(
            "Détection auto de l'expérience",
            value=True,
            help="Extrait automatiquement l'expérience de la requête (ex: 'senior' = 7 ans)"
        )

        if not use_auto_experience:
            min_exp = st.slider(
                "Expérience minimum (années)",
                min_value=0,
                max_value=15,
                value=0,
                step=1
            )
        else:
            min_exp = None

        max_results = st.slider(
            "Nombre de candidats",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

        st.divider()

        # Options avancées
        with st.expander("Options avancées"):
            show_scores = st.checkbox("Afficher les scores", value=True)
            show_chunks = st.checkbox("Afficher les chunks", value=False)
            show_competences = st.checkbox("Afficher toutes les compétences", value=True)

        st.divider()

        # Exemples de requêtes
        st.subheader("💡 Exemples de requêtes")

        example_queries = [
            "Data Scientist avec Python et ML",
            "Développeur React senior",
            "Ingénieur DevOps Kubernetes",
            "Architecte cloud AWS 8+ ans",
            "Expert cybersécurité CISSP"
        ]

        for example in example_queries:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.query = example

    # Zone principale
    col1, col2 = st.columns([2, 1])

    with col1:
        # Input utilisateur
        query = st.text_input(
            "🔍 Décrivez le profil recherché",
            value=st.session_state.get('query', ''),
            placeholder="Ex: Data Scientist avec 7+ ans d'expérience en machine learning",
            key="query_input"
        )

        # Bouton de recherche
        search_button = st.button("🚀 Rechercher", type="primary", use_container_width=True)

    with col2:
        # Statistiques de la dernière recherche
        if 'last_search_time' in st.session_state:
            st.metric("Temps de recherche", f"{st.session_state.last_search_time:.2f}s")
            st.metric("Candidats trouvés", st.session_state.get('candidates_count', 0))

    st.divider()

    # Traitement de la recherche
    if search_button and query:
        with st.spinner("Recherche en cours..."):
            start_time = time.time()

            # 1. Classification
            is_hr = classify_query_with_llm(query, groq_client)

            if not is_hr:
                st.warning("⚠️ Cette question ne semble pas liée au recrutement.")
                st.info("""Je suis spécialisé dans la recherche de candidats et ne peux répondre qu'aux questions RH concernant:
- La recherche de profils
- Les compétences techniques
- L'expérience professionnelle
- Les recommandations de recrutement""")
                st.stop()

            # 2. Extraction expérience
            if use_auto_experience:
                min_exp = extract_min_experience(query)
                if min_exp > 0:
                    st.info(f"📊 Expérience minimum détectée: {min_exp} ans")

            # 3. Recherche
            candidates = search_candidates(
                query, es, embedding_model,
                min_experience=min_exp or 0,
                max_results=max_results
            )

            # 4. Génération réponse LLM
            if candidates:
                response = generate_response(query, candidates, groq_client)
            else:
                response = "Aucun candidat pertinent trouvé pour cette requête. Essayez d'élargir vos critères."

            search_time = time.time() - start_time

            # Sauvegarde en session
            st.session_state.last_search_time = search_time
            st.session_state.candidates_count = len(candidates)
            st.session_state.last_candidates = candidates
            st.session_state.last_response = response

    # Affichage des résultats (reste identique)
    if 'last_response' in st.session_state:
        st.header("📋 Recommandation")
        st.markdown(st.session_state.last_response)

        st.divider()

        # Détails des candidats
        if st.session_state.get('last_candidates'):
            st.header("👥 Détails des candidats")

            candidates = st.session_state.last_candidates

            # Séparation pertinents / moins pertinents
            highly_relevant = [c for c in candidates if c['_score'] >= 10.0]
            less_relevant = [c for c in candidates if c['_score'] < 10.0]

            if highly_relevant:
                st.subheader(f"✅ Très pertinents ({len(highly_relevant)})")

                for i, hit in enumerate(highly_relevant, 1):
                    doc = hit['_source']

                    with st.expander(
                            f"{i}. {doc['candidat_nom']} - {doc.get('candidat_poste', 'N/A')} "
                            f"({'⭐' * min(int(hit['_score'] / 5), 5)})",
                            expanded=(i == 1)
                    ):
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric("Expérience", f"{doc.get('candidat_experience', 'N/A')} ans")

                        with col_b:
                            if show_scores:
                                st.metric("Score", f"{hit['_score']:.2f}")

                        with col_c:
                            st.metric("Type", doc['chunk_type'])

                        # Compétences
                        if doc.get('competences_liste') and show_competences:
                            st.markdown("**Compétences:**")
                            competences = doc['competences_liste']

                            # Affichage en badges
                            comp_html = " ".join([
                                f'<span style="background-color: #0066cc; color: white; padding: 4px 8px; margin: 2px; border-radius: 4px; display: inline-block; font-size: 12px;">{comp}</span>'
                                for comp in competences[:20]
                            ])
                            st.markdown(comp_html, unsafe_allow_html=True)

                        # Chunk complet
                        if show_chunks:
                            st.markdown("**Contexte complet:**")
                            st.text(doc['chunk_text'])

            if less_relevant:
                st.subheader(f"⚠️ Moins pertinents ({len(less_relevant)})")

                with st.expander("Voir les candidats moins pertinents"):
                    for i, hit in enumerate(less_relevant, 1):
                        doc = hit['_source']
                        st.markdown(
                            f"**{i}. {doc['candidat_nom']}** - {doc.get('candidat_poste', 'N/A')} (score: {hit['_score']:.2f})")

        # Actions
        st.divider()

        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            if st.button("🔄 Nouvelle recherche", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        with col_action2:
            # Export JSON
            if st.session_state.get('last_candidates'):
                import json
                export_data = {
                    "query": query,
                    "date": datetime.now().isoformat(),
                    "candidates": [
                        {
                            "nom": hit['_source']['candidat_nom'],
                            "poste": hit['_source'].get('candidat_poste', 'N/A'),
                            "experience": hit['_source'].get('candidat_experience', 0),
                            "score": hit['_score'],
                            "competences": hit['_source'].get('competences_liste', [])[:10]
                        }
                        for hit in st.session_state.last_candidates
                    ]
                }

                st.download_button(
                    "📥 Télécharger (JSON)",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"candidats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        with col_action3:
            # Export texte
            if st.session_state.get('last_response'):
                export_text = f"""RECHERCHE DE CANDIDATS
======================

Requête: {query}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

RECOMMANDATION
==============

{st.session_state.last_response}

CANDIDATS
=========

"""
                for i, hit in enumerate(st.session_state.last_candidates, 1):
                    doc = hit['_source']
                    export_text += f"{i}. {doc['candidat_nom']} - {doc.get('candidat_poste', 'N/A')}\n"
                    export_text += f"   Expérience: {doc.get('candidat_experience', 'N/A')} ans\n"
                    export_text += f"   Score: {hit['_score']:.2f}\n\n"

                st.download_button(
                    "📄 Télécharger (TXT)",
                    data=export_text,
                    file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Assistant RH v2.0 Cloud | Propulsé par Elasticsearch Cloud + BGE-M3 + Llama 3.3 70B (Groq) | 30 CVs indexés
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    # Vérifications
    if not GROQ_API_KEY:
        st.error("⚠️ Variable d'environnement GROQ_API_KEY non définie")
        st.info("Configurez GROQ_API_KEY dans les secrets Streamlit")
        st.stop()

    if not ES_CLOUD_URL or not ES_API_KEY:
        st.error("⚠️ Configuration Elastic Cloud manquante")
        st.info("Configurez ES_CLOUD_URL et ES_API_KEY dans les secrets Streamlit")
        st.stop()

    main()