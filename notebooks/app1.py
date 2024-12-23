import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report

# Configuration
st.set_page_config(layout="wide")

# Chemins
BASE_DIR = Path(__file__).parent.parent
MODELS = {
   "Régression Logistique": "logistic_regression.pkl",
   "SVM": "svm_model.pkl"
}
VECTORIZER_PATH = BASE_DIR / "models/classical_m/tfidf_vectorizer.pkl"

try:
   # Chargement du vectorizer
   vectorizer = joblib.load(VECTORIZER_PATH)
   
   # Interface
   st.title("📊 Classification d'Avis Clients")
   
   # Sélection du modèle
   selected_model = st.sidebar.selectbox(
       "Choisir le modèle",
       list(MODELS.keys())
   )
   
   # Chargement du modèle sélectionné
   model_path = BASE_DIR / f"models/classical_m/{MODELS[selected_model]}"
   model = joblib.load(model_path)
   
   # Zone de saisie
   example_review = st.text_area(
       "Entrez un avis client",
       placeholder="Tapez votre avis ici...",
       height=150
   )
   
   if example_review:
       # Prédiction
       example_review_vectorized = vectorizer.transform([example_review])
       prediction = model.predict(example_review_vectorized)
       proba = model.predict_proba(example_review_vectorized)[0]
       
       # Affichage résultats
       col1, col2 = st.columns(2)
       with col1:
           st.markdown(f"### Prédiction: {'🟢 Positif' if prediction[0] == 1 else '🔴 Négatif'}")
           st.markdown(f"### Confiance: {max(proba) * 100:.1f}%")
           
       with col2:
           # Graphique confiance
           fig = go.Figure(go.Bar(
               x=['Négatif', 'Positif'],
               y=[proba[0]*100, proba[1]*100],
               marker_color=['#ff9999', '#99ff99']
           ))
           fig.update_layout(
               title="Probabilités de classification",
               yaxis_title="Pourcentage (%)"
           )
           st.plotly_chart(fig)

   # Métriques et graphiques
   if st.button("📈 Analyser les performances"):
       # Données fictives pour démonstration
       metrics = {
           'Précision': 0.85,
           'Rappel': 0.82,
           'F1-Score': 0.83
       }
       
       # Graphique radar
       fig = go.Figure()
       fig.add_trace(go.Scatterpolar(
           r=list(metrics.values()),
           theta=list(metrics.keys()),
           fill='toself'
       ))
       fig.update_layout(
           polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
           title="Métriques de performance"
       )
       st.plotly_chart(fig)

except FileNotFoundError as e:
   st.error(f"❌ Erreur: Fichier non trouvé\n{str(e)}")
except Exception as e:
   st.error(f"❌ Une erreur est survenue: {str(e)}")

# Pied de page
st.markdown("---")
st.markdown("*Développé avec Streamlit et Scikit-learn*")
