import os
import joblib
import streamlit as st
from textblob import TextBlob
import numpy as np
import plotly.express as px
import numpy as np

# Couleurs pour les graphiques
SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral': '#f1c40f',
    'negative': '#e74c3c'
}

class SentimentAnalyzer:
    def __init__(self, models_path):
        self.models_path = models_path
        self.models = {}
        self.vectorizer = None
        self.history = []

    def setup_models(self):
        """
        Charge les modèles de classification et le vectorizer.
        """
        try:
            self.vectorizer = joblib.load(os.path.join(self.models_path, "tfidf_vectorizer.pkl"))
            self.models['logistic_regression'] = joblib.load(os.path.join(self.models_path, "logistic_regression.pkl"))
            self.models['svm'] = joblib.load(os.path.join(self.models_path, "svm_model.pkl"))
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {e}")

    def analyze_with_textblob(self, text):
        """Analyse le sentiment en utilisant TextBlob."""
        try:
            # Assurez-vous que le texte est une chaîne de caractères
            if isinstance(text, str):
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return "positive", [0.7, 0.2, 0.1]
                elif polarity < -0.1:
                    return "negative", [0.1, 0.2, 0.7]
                else:
                    return "neutral", [0.2, 0.6, 0.2]
            else:
                st.error("TextBlob ne peut analyser que des chaînes de texte.")
                return "unknown", [0.0, 0.0, 0.0]
        except Exception as e:
            st.error(f"Erreur TextBlob : {e}")
            return "unknown", [0.0, 0.0, 0.0]

    def predict_sentiment(self, text, model_name):
        """Prédit le sentiment à l'aide d'un modèle de machine learning."""
        if model_name not in self.models:
            st.error(f"Modèle {model_name} non trouvé.")
            return None, None

        if not self.vectorizer:
            st.error("Vectorizer non chargé !")
            return None, None

        try:
            model = self.models[model_name]
            # Transformer le texte en vecteur TF-IDF
            if isinstance(text, str):
                text_tfidf = self.vectorizer.transform([text])
            elif isinstance(text, np.ndarray):
                text_tfidf = text
            else:
                st.error("Erreur : le texte doit être une chaîne de caractères ou un tableau numpy.")
                return None, None

            # Prédiction et probabilités
            prediction = model.predict(text_tfidf)
            probabilities = model.predict_proba(text_tfidf)[0]
            
            # Convertir la prédiction en label de sentiment
            sentiment_labels = {
                0: "negative",
                1: "neutral",
                2: "positive"
            }
            sentiment = sentiment_labels.get(prediction[0], "unknown")
            
            return sentiment, probabilities
        except Exception as e:
            st.error(f"Erreur prédiction : {e}")
            return None, None

    def log_history(self, text, sentiment, probabilities):
        """Enregistre l'analyse dans l'historique."""
        self.history.append({
            'text': text,
            'sentiment': sentiment,
            'probabilities': probabilities
        })

# Fonctions pour créer des graphiques
def create_sentiment_distribution_chart(history):
    """Crée un graphique de distribution des sentiments."""
    sentiments = [entry['sentiment'] for entry in history if entry['sentiment'] != 'unknown']
    if not sentiments:
        return None
    fig = px.histogram(
        x=sentiments,
        title="Distribution des sentiments",
        color=sentiments,
        color_discrete_map=SENTIMENT_COLORS
    )
    return fig

def create_sentiment_probabilities_chart(history):
    """Crée un graphique des probabilités de sentiments."""
    import pandas as pd
    import plotly.express as px
    
    if not history:
        return None
    
    # Extraire les probabilités pour chaque sentiment
    proba_data = []
    for entry in history:
        if entry['sentiment'] != 'unknown':
            proba_data.append({
                'Sentiment': entry['sentiment'],
                'Negative': entry['probabilities'][0],
                'Neutral': entry['probabilities'][1],
                'Positive': entry['probabilities'][2]
            })
    
    if not proba_data:
        return None
    
    # Créer un DataFrame pandas
    df = pd.DataFrame(proba_data)
    
    # Convertir le DataFrame en format long pour plotly
    df_melted = df.melt(
        id_vars=['Sentiment'], 
        var_name='Probability Type', 
        value_name='Probability'
    )
    
    # Créer un graphique à barres groupées
    fig = px.bar(
        df_melted, 
        x='Sentiment', 
        y='Probability', 
        color='Probability Type',
        title='Probabilités des Sentiments',
        labels={'Probability': 'Probabilité'},
        color_discrete_map={
            'Negative': SENTIMENT_COLORS['negative'],
            'Neutral': SENTIMENT_COLORS['neutral'],
            'Positive': SENTIMENT_COLORS['positive']
        }
    )
    
    return fig

def main():
    st.title("Analyseur de Sentiments - Customer Reviews")
    st.write("Cette application utilise des modèles de Machine Learning pour analyser les sentiments des avis clients.")

    # Chemin vers les modèles (à ajuster selon votre configuration)
    models_path = r"../models/classical_m"  # Chemin relatif recommandé
    
    # Vérifier si le chemin existe
    if not os.path.exists(models_path):
        st.error(f"Le chemin des modèles n'existe pas : {models_path}")
        return

    analyzer = SentimentAnalyzer(models_path)
    analyzer.setup_models()

    # Entrée utilisateur
    text_input = st.text_area("Entrez un texte ou un avis client ici")
    model_choice = st.selectbox("Choisissez un modèle", ["TextBlob", "Logistic Regression", "SVM"])
    show_charts = st.checkbox("Afficher les graphiques d'analyse")

    if st.button("Analyser"):
        if not text_input:
            st.warning("Veuillez entrer un texte.")
        else:
            # Analyse du texte avec le modèle choisi
            if model_choice == "TextBlob":
                sentiment, probabilities = analyzer.analyze_with_textblob(text_input)
            elif model_choice == "Logistic Regression":
                sentiment, probabilities = analyzer.predict_sentiment(text_input, "logistic_regression")
            elif model_choice == "SVM":
                sentiment, probabilities = analyzer.predict_sentiment(text_input, "svm")
            else:
                sentiment, probabilities = "unknown", [0.0, 0.0, 0.0]

            # Afficher le résultat
            if sentiment:
                st.success(f"Sentiment prédit : {sentiment.capitalize()}")
                st.write(f"Probabilités : {np.round(probabilities, 2)}")
                analyzer.log_history(text_input, sentiment, probabilities)

            # Afficher les graphiques
            if show_charts:
                # Distribution des sentiments
                sentiment_chart = create_sentiment_distribution_chart(analyzer.history)
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                
                # Probabilités des sentiments
                probabilities_chart = create_sentiment_probabilities_chart(analyzer.history)
                if probabilities_chart:
                    st.plotly_chart(probabilities_chart, use_container_width=True)

if __name__ == "__main__":
    main()