import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from datetime import datetime
import altair as alt

# Configuration
st.set_page_config(layout="wide", page_title="Customer Review Classification", page_icon="🖊")

# Style personnalisé
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest.pkl"
}
VECTORIZER_PATH = BASE_DIR / "models/classical_m/tfidf_vectorizer.pkl"

def plot_confidence_distribution(probabilities):
    fig = go.Figure()
    
    categories = ['Négatif', 'Neutre', 'Positif']
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    
    for i, (cat, prob, color) in enumerate(zip(categories, probabilities, colors)):
        fig.add_trace(go.Bar(
            name=cat,
            y=[cat],
            x=[prob * 100],
            orientation='h',
            marker_color=color,
            text=f'{prob * 100:.1f}%',
            textposition='auto',
        ))

    fig.update_layout(
        title="Distribution des Probabilités",
        xaxis_title="Probabilité (%)",
        yaxis_title="Sentiment",
        barmode='group',
        height=250,
        showlegend=False
    )
    return fig

def analyze_review_length(review):
    words = len(review.split())
    chars = len(review)
    sentences = len([s for s in review.split('.') if s.strip()])
    
    return pd.DataFrame({
        'Métrique': ['Mots', 'Caractères', 'Phrases'],
        'Valeur': [words, chars, sentences]
    })

def track_predictions(prediction, confidence):
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'prediction': prediction,
        'confidence': confidence
    })

try:
    # Load vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Interface
    st.title("🖊 Classification des Avis Clients")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Prédiction", "Analyse en Masse", "Statistiques"])
    
    with tab1:
        # Model selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model = st.selectbox(
                "Choisir un modèle",
                list(MODELS.keys())
            )
            
        with col2:
            language = st.selectbox(
                "Langue",
                ["Français", "English", "Español"]
            )

        # Load selected model
        model_path = BASE_DIR / f"models/classical_m/{MODELS[selected_model]}"
        model = joblib.load(model_path)

        # Input area
        example_review = st.text_area(
            "Entrez un avis client",
            placeholder="Tapez votre avis ici...",
            height=150
        )

        if example_review:
            # Text analysis
            length_stats = analyze_review_length(example_review)
            
            # Prediction
            example_review_vectorized = vectorizer.transform([example_review])
            prediction = model.predict(example_review_vectorized)
            proba = model.predict_proba(example_review_vectorized)[0]
            
            # Track prediction
            track_predictions(prediction[0], max(proba))

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                sentiment = "🟢 Positif" if prediction[0] == 1 else ("🔴 Négatif" if prediction[0] == 0 else "🟡 Neutre")
                st.markdown(f"### Prédiction: {sentiment}")
                st.markdown(f"### Confiance: {max(proba) * 100:.1f}%")
                
                # Statistiques du texte
                st.markdown("### Analyse du texte")
                st.dataframe(length_stats, hide_index=True)

            with col2:
                st.plotly_chart(plot_confidence_distribution(proba))

    with tab2:
        st.subheader("Prédiction en Masse")
        uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'review' in data.columns:
                progress_bar = st.progress(0)
                predictions = []
                probabilities = []
                
                for i, review in enumerate(data['review']):
                    pred = model.predict(vectorizer.transform([review]))
                    prob = model.predict_proba(vectorizer.transform([review]))
                    predictions.append(pred[0])
                    probabilities.append(prob[0])
                    progress_bar.progress((i + 1) / len(data))
                
                data['prediction'] = predictions
                data['confidence'] = [max(p) * 100 for p in probabilities]
                
                # Visualisations
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(data, names='prediction', title='Distribution des Prédictions')
                    st.plotly_chart(fig)
                
                with col2:
                    fig = px.histogram(data, x='confidence', title='Distribution des Niveaux de Confiance')
                    st.plotly_chart(fig)
                
                st.dataframe(data)
                st.download_button("Télécharger les Résultats", data.to_csv(index=False), "resultats.csv", "text/csv")
            else:
                st.error("Le fichier CSV doit contenir une colonne 'review'.")

    with tab3:
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            st.subheader("Historique des Prédictions")
            
            # Graphique temporel
            fig = px.line(history_df, x='timestamp', y='confidence', title='Évolution de la Confiance')
            st.plotly_chart(fig)
            
            # Distribution des prédictions
            pred_counts = history_df['prediction'].value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                        title='Distribution des Prédictions')
            st.plotly_chart(fig)
        else:
            st.info("Aucun historique disponible. Commencez à faire des prédictions!")

except FileNotFoundError as e:
    st.error(f"❌ Erreur: Fichier non trouvé\n{str(e)}")
except Exception as e:
    st.error(f"❌ Une erreur s'est produite: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Développé par ENIAD*")
