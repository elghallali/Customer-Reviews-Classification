import os
import joblib
import streamlit as st
from textblob import TextBlob
import numpy as np
import plotly.express as px

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
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "positive", [0.7, 0.2, 0.1]
        elif polarity < -0.1:
            return "negative", [0.1, 0.2, 0.7]
        else:
            return "neutral", [0.2, 0.6, 0.2]

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
            text_tfidf = self.vectorizer.transform([text])
            prediction = model.predict(text_tfidf)
            probabilities = model.predict_proba(text_tfidf)[0]

            sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
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

# Application principale
def main():
    st.title("Analyseur de Sentiments - Customer Reviews")
    st.write("Cette application utilise des modèles de Machine Learning pour analyser les sentiments des avis clients.")

    # Chemin vers les modèles
    models_path = "./models/classical_ml"

    if not os.path.exists(models_path):
        st.error(f"Le chemin des modèles n'existe pas : {models_path}")
        return

    analyzer = SentimentAnalyzer(models_path)
    analyzer.setup_models()

    text_input = st.text_area("Entrez un texte ou un avis client ici")
    model_choice = st.selectbox("Choisissez un modèle", ["TextBlob", "Logistic Regression", "SVM"])
    show_charts = st.checkbox("Afficher les graphiques d'analyse")

    if st.button("Analyser"):
        if not text_input:
            st.warning("Veuillez entrer un texte.")
        else:
            if model_choice == "TextBlob":
                sentiment, probabilities = analyzer.analyze_with_textblob(text_input)
            elif model_choice == "Logistic Regression":
                sentiment, probabilities = analyzer.predict_sentiment(text_input, "logistic_regression")
            elif model_choice == "SVM":
                sentiment, probabilities = analyzer.predict_sentiment(text_input, "svm")
            else:
                sentiment, probabilities = "unknown", [0.0, 0.0, 0.0]

            if sentiment:
                st.success(f"Sentiment prédit : {sentiment.capitalize()}")
                st.write(f"Probabilités : {np.round(probabilities, 2)}")
                analyzer.log_history(text_input, sentiment, probabilities)

if __name__ == "__main__":
    main()
