import streamlit as st
import joblib
import os
import numpy as np
from scipy.sparse import csr_matrix

# Configuration de la page en premier
st.set_page_config(page_title="Customer Reviews Sentiment Analysis", layout="wide")

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Le fichier {model_path} n'a pas été trouvé. Erreur: {str(e)}")
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
        text_tfidf = vectorizer.transform([text])
        text_tfidf_aligned = align_features(text_tfidf, expected_features)
        prediction = model.predict(text_tfidf_aligned)
        
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(text_tfidf_aligned)[0]
        else:
            prediction_proba = np.zeros(3)
            prediction_proba[prediction[0]] = 1.0
        
        sentiment_label = get_sentiment_label(prediction[0])
        return sentiment_label, prediction_proba
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None, None

# Définir les chemins des modèles
models_path = "/mount/src/customer-reviews-classification/models/classical_m"
if not os.path.exists(models_path):
    # Essayer un chemin relatif si le chemin absolu ne fonctionne pas
    models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "classical_m")

# Afficher le chemin pour le débogage
st.write(f"Chemin des modèles: {models_path}")

# Charger les modèles
logistic_regression_path = os.path.join(models_path, "logistic_regression.pkl")
svm_path = os.path.join(models_path, "svm_model.pkl")
vectorizer_path = os.path.join(models_path, "tfidf_vectorizer.pkl")

# Vérifier l'existence des fichiers
for path in [logistic_regression_path, svm_path, vectorizer_path]:
    if not os.path.exists(path):
        st.error(f"Fichier non trouvé: {path}")

logistic_regression_model = load_model(logistic_regression_path)
svm_model = load_model(svm_path)
tfidf_vectorizer = load_model(vectorizer_path)

# Interface Streamlit
st.title("Customer Reviews Sentiment Analysis")

if not all([logistic_regression_model, svm_model, tfidf_vectorizer]):
    st.error("Certains modèles n'ont pas pu être chargés. Vérifiez que tous les fichiers .pkl sont présents dans le dossier models/classical_m/")
    st.stop()

st.sidebar.header("Choisir un modèle")
model_choice = st.sidebar.selectbox("Sélectionnez un modèle", ["Logistic Regression", "SVM"])

st.sidebar.header("Entrer un texte")
user_input = st.sidebar.text_area(
    "Entrez votre texte ici",
    "The service was excellent and the food was delicious. I had a wonderful time and will definitely come back!"
)

if st.sidebar.button("Prédire"):
    model = logistic_regression_model if model_choice == "Logistic Regression" else svm_model
    
    sentiment_label, prediction_proba = predict_sentiment(model, user_input, tfidf_vectorizer)
    
    if sentiment_label is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### Texte analysé")
            st.write(user_input)
            
            st.write("### Résultats de l'analyse")
            st.write(f"**Sentiment prédit :** {sentiment_label}")
        
        with col2:
            st.write("### Probabilités par classe")
            sentiment_labels = ['positive', 'neutral', 'negative']
            for label, prob in zip(sentiment_labels, prediction_proba):
                st.write(f"**{label.capitalize()}**")
                st.progress(float(prob))
                st.write(f"{prob*100:.2f}%")

        if sentiment_label == "positive":
            st.success("✨ Cette review est positive !")
        elif sentiment_label == "negative":
            st.error("😟 Cette review est négative.")
        else:
            st.info("😐 Cette review est neutre.")

        if model_choice == "SVM" and not hasattr(model, 'predict_proba'):
            st.info("Note : Le modèle SVM actuel ne fournit pas de probabilités.")

# Informations sur le projet
st.sidebar.markdown("---")
st.sidebar.header("À propos")
st.sidebar.info("""
Cette application utilise le Machine Learning pour analyser le sentiment des avis clients.
- **Modèles disponibles :** Régression Logistique et SVM
- **Classes de sentiment :** Positif, Neutre, Négatif
""")