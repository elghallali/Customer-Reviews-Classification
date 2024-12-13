import streamlit as st
import joblib
import os
import numpy as np
from scipy.sparse import csr_matrix

# Configuration de la page en premier
st.set_page_config(page_title="Customer Reviews Sentiment Analysis", layout="wide")

def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Le fichier {model_path} n'a pas été trouvé.")
        return None

def align_features(text_tfidf, expected_features):
    if text_tfidf.shape[1] < expected_features:
        aligned_features = csr_matrix((
            text_tfidf.data,
            text_tfidf.indices,
            text_tfidf.indptr
        ), shape=(text_tfidf.shape[0], expected_features))
        return aligned_features
    return text_tfidf

def get_sentiment_label(prediction):
    sentiment_mapping = {
        0: "positive",
        1: "neutral",
        2: "negative"
    }
    return sentiment_mapping.get(prediction, "unknown")

def predict_sentiment(model, text, vectorizer, expected_features=59732):
    if model is None or vectorizer is None:
        st.error("Le modèle ou le vectoriseur n'a pas été chargé correctement.")
        return None, None
    
    try:
        # Vectorisation du texte
        text_tfidf = vectorizer.transform([text])
        
        # Alignement des features
        text_tfidf_aligned = align_features(text_tfidf, expected_features)
        
        # Prédiction
        prediction = model.predict(text_tfidf_aligned)
        
        # Gestion différente pour SVM et Régression Logistique
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(text_tfidf_aligned)[0]
        else:
            prediction_proba = np.zeros(3)  # 3 classes: positive, neutral, negative
            prediction_proba[prediction[0]] = 1.0
        
        # Convertir la prédiction numérique en label
        sentiment_label = get_sentiment_label(prediction[0])
        
        return sentiment_label, prediction_proba
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None, None

# Définir les chemins des modèles
current_dir = os.path.dirname(os.path.abspath(__file__))
models_base_path = os.path.join(current_dir, 'models')

# Charger les modèles
logistic_regression_model = load_model(os.path.join(models_base_path, 'logistic_regression.pkl'))
svm_model = load_model(os.path.join(models_base_path, 'svm_model.pkl'))
tfidf_vectorizer = load_model(os.path.join(models_base_path, 'tfidf_vectorizer.pkl'))

# Interface Streamlit
st.title("Customer Reviews Sentiment Analysis")

st.sidebar.header("Choisir un modèle")
model_choice = st.sidebar.selectbox("Sélectionnez un modèle", ["Logistic Regression", "SVM"])

st.sidebar.header("Entrer un texte")
user_input = st.sidebar.text_area(
    "Entrez votre texte ici",
    "The service was excellent and the food was delicious. I had a wonderful time and will definitely come back!"
)

if st.sidebar.button("Prédire"):
    model = logistic_regression_model if model_choice == "Logistic Regression" else svm_model
    
    if model is None:
        st.error("Le modèle n'a pas pu être chargé.")
    else:
        sentiment_label, prediction_proba = predict_sentiment(model, user_input, tfidf_vectorizer)
        
        if sentiment_label is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### Texte analysé")
                st.write(user_input)
                
                st.write("### Résultats de l'analyse")
                st.write(f"**Sentiment prédit :** {sentiment_label}")
            
            with col2:
                # Affichage des probabilités avec des barres de progression
                st.write("### Probabilités par classe")
                sentiment_labels = ['positive', 'neutral', 'negative']
                for label, prob in zip(sentiment_labels, prediction_proba):
                    st.write(f"**{label.capitalize()}**")
                    st.progress(float(prob))
                    st.write(f"{prob*100:.2f}%")

            # Afficher un message d'explication
            if sentiment_label == "positive":
                st.success("✨ Cette review est positive !")
            elif sentiment_label == "negative":
                st.error("😟 Cette review est négative.")
            else:
                st.info("😐 Cette review est neutre.")

            # Si c'est un SVM, ajouter une note explicative
            if model_choice == "SVM" and not hasattr(model, 'predict_proba'):
                st.info("Note : Le modèle SVM actuel ne fournit pas de probabilités. Les valeurs affichées représentent uniquement la classe prédite.")

# Ajouter des informations sur le projet
st.sidebar.markdown("---")
st.sidebar.header("À propos")
st.sidebar.info("""
Cette application utilise le Machine Learning pour analyser le sentiment des avis clients.
- **Modèles disponibles :** Régression Logistique et SVM
- **Classes de sentiment :** Positif, Neutre, Négatif
""")
