import os
import joblib
import streamlit as st
from textblob import TextBlob
import numpy as np
import plotly.express as px
import pandas as pd

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyseur de Sentiments",
    page_icon="🎭",
    layout="wide"
)

# Couleurs pour les graphiques
SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral': '#f1c40f',
    'negative': '#e74c3c'
}

# Chemins des modèles
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "classical_ml")

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.history = []
        self.setup_models()

    def setup_models(self):
        """Charge les modèles de classification et le vectorizer."""
        try:
            # Chemins des modèles
            vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
            lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
            svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")

            # Vérification de l'existence des fichiers
            if not all(os.path.exists(path) for path in [vectorizer_path, lr_path, svm_path]):
                st.error(f"Certains fichiers modèles sont manquants dans : {MODELS_DIR}")
                st.error(f"Fichiers disponibles : {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Dossier non trouvé'}")
                return

            # Chargement des modèles
            self.vectorizer = joblib.load(vectorizer_path)
            self.models['logistic_regression'] = joblib.load(lr_path)
            self.models['svm'] = joblib.load(svm_path)

        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {str(e)}")

    def analyze_with_textblob(self, text):
        """Analyse le sentiment en utilisant TextBlob."""
        try:
            if not isinstance(text, str):
                raise ValueError("Le texte doit être une chaîne de caractères")
                
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return "positive", [0.1, 0.2, 0.7]
            elif polarity < -0.1:
                return "negative", [0.7, 0.2, 0.1]
            else:
                return "neutral", [0.2, 0.6, 0.2]
                
        except Exception as e:
            st.error(f"Erreur TextBlob : {str(e)}")
            return "unknown", [0.0, 0.0, 0.0]

    def predict_sentiment(self, text, model_name):
        """Prédit le sentiment à l'aide d'un modèle de machine learning."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Modèle {model_name} non trouvé")

            if not self.vectorizer:
                raise ValueError("Vectorizer non chargé")

            # Vectorisation du texte
            text_vectorized = self.vectorizer.transform([text])
            
            # Prédiction
            model = self.models[model_name]
            prediction = model.predict(text_vectorized)
            probabilities = model.predict_proba(text_vectorized)[0]
            
            # Conversion de la prédiction en label
            sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_labels.get(prediction[0], "unknown")
            
            return sentiment, probabilities
            
        except Exception as e:
            st.error(f"Erreur de prédiction : {str(e)}")
            return None, None

    def log_history(self, text, sentiment, probabilities):
        """Enregistre l'analyse dans l'historique."""
        if sentiment and probabilities is not None:
            self.history.append({
                'text': text,
                'sentiment': sentiment,
                'probabilities': probabilities
            })

def create_sentiment_distribution_chart(history):
    """Crée un graphique de distribution des sentiments."""
    if not history:
        return None
        
    sentiments = [entry['sentiment'] for entry in history if entry['sentiment'] != 'unknown']
    if not sentiments:
        return None
        
    fig = px.histogram(
        x=sentiments,
        title="Distribution des Sentiments",
        color=sentiments,
        color_discrete_map=SENTIMENT_COLORS,
        labels={'x': 'Sentiment', 'count': 'Nombre d\'analyses'}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Sentiment",
        yaxis_title="Nombre d'analyses"
    )
    
    return fig

def create_sentiment_probabilities_chart(history):
    """Crée un graphique des probabilités de sentiments."""
    if not history:
        return None
    
    proba_data = []
    for entry in history:
        if entry['sentiment'] != 'unknown':
            proba_data.append({
                'Sentiment': entry['sentiment'],
                'Négatif': entry['probabilities'][0],
                'Neutre': entry['probabilities'][1],
                'Positif': entry['probabilities'][2]
            })
    
    if not proba_data:
        return None
    
    df = pd.DataFrame(proba_data)
    df_melted = df.melt(
        id_vars=['Sentiment'],
        var_name='Type de Probabilité',
        value_name='Probabilité'
    )
    
    fig = px.bar(
        df_melted,
        x='Sentiment',
        y='Probabilité',
        color='Type de Probabilité',
        title='Probabilités des Sentiments par Catégorie',
        color_discrete_map={
            'Négatif': SENTIMENT_COLORS['negative'],
            'Neutre': SENTIMENT_COLORS['neutral'],
            'Positif': SENTIMENT_COLORS['positive']
        }
    )
    
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Probabilité"
    )
    
    return fig

def main():
    # Titre et description
    st.title("🎭 Analyseur de Sentiments - Customer Reviews")
    st.markdown("""
    Cette application utilise des modèles de Machine Learning pour analyser les sentiments des avis clients.
    Vous pouvez choisir entre différents modèles d'analyse et visualiser les résultats.
    """)

    # Initialisation de l'analyseur
    analyzer = SentimentAnalyzer()

    # Interface utilisateur
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Entrez votre texte ici",
                height=150,
                placeholder="Tapez ou collez votre texte ici..."
            )
        
        with col2:
            model_choice = st.selectbox(
                "Choisissez un modèle",
                ["TextBlob", "Logistic Regression", "SVM"]
            )
            show_charts = st.checkbox("Afficher les graphiques d'analyse", value=True)
            
            if st.button("Analyser", type="primary"):
                if not text_input:
                    st.warning("⚠️ Veuillez entrer un texte à analyser.")
                else:
                    with st.spinner("Analyse en cours..."):
                        # Analyse selon le modèle choisi
                        if model_choice == "TextBlob":
                            sentiment, probabilities = analyzer.analyze_with_textblob(text_input)
                        elif model_choice == "Logistic Regression":
                            sentiment, probabilities = analyzer.predict_sentiment(text_input, "logistic_regression")
                        else:  # SVM
                            sentiment, probabilities = analyzer.predict_sentiment(text_input, "svm")

                        # Affichage des résultats
                        if sentiment and probabilities is not None:
                            # Enregistrement dans l'historique
                            analyzer.log_history(text_input, sentiment, probabilities)
                            
                            # Affichage du résultat
                            sentiment_color = SENTIMENT_COLORS.get(sentiment, '#000000')
                            st.markdown(f"""
                            ### Résultat de l'analyse
                            - **Sentiment détecté:** <span style='color:{sentiment_color}'>{sentiment.upper()}</span>
                            - **Probabilités:**
                                - Négatif: {probabilities[0]:.2%}
                                - Neutre: {probabilities[1]:.2%}
                                - Positif: {probabilities[2]:.2%}
                            """, unsafe_allow_html=True)

    # Affichage des graphiques
    if show_charts and analyzer.history:
        st.markdown("### Visualisation des Résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            dist_chart = create_sentiment_distribution_chart(analyzer.history)
            if dist_chart:
                st.plotly_chart(dist_chart, use_container_width=True)
                
        with col2:
            prob_chart = create_sentiment_probabilities_chart(analyzer.history)
            if prob_chart:
                st.plotly_chart(prob_chart, use_container_width=True)

if __name__ == "__main__":
    main()
