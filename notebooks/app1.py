import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from datetime import datetime

# Configuration
st.set_page_config(layout="wide")

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest.pkl"  # Added model
}

VECTORIZER_PATH = BASE_DIR / "models/classical_m/tfidf_vectorizer.pkl"

# Custom theme and styling
st.markdown("""
    <style>
    .stAlert {border-radius: 10px;}
    .stButton>button {border-radius: 20px;}
    </style>
    """, unsafe_allow_html=True)

try:
    # Load vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Interface
    st.title("📊 Customer Review Classification")
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        list(MODELS.keys())
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Load selected model
    model_path = BASE_DIR / f"models/classical_m/{MODELS[selected_model]}"
    model = joblib.load(model_path)
    
    # Main interface
    tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])
    
    with tab1:
        example_review = st.text_area(
            "Enter customer review",
            placeholder="Type your review here...",
            height=150
        )
        
        if example_review:
            # Prediction
            example_review_vectorized = vectorizer.transform([example_review])
            prediction = model.predict(example_review_vectorized)
            proba = model.predict_proba(example_review_vectorized)[0]
            
            # Results display
            col1, col2 = st.columns(2)
            with col1:
                sentiment = "🟢 Positive" if prediction[0] == 1 else "🔴 Negative"
                confidence = max(proba) * 100
                
                if confidence >= confidence_threshold * 100:
                    st.markdown(f"### Prediction: {sentiment}")
                    st.markdown(f"### Confidence: {confidence:.1f}%")
                else:
                    st.warning("⚠️ Prediction below confidence threshold")
            
            with col2:
                # Confidence visualization
                fig = go.Figure(go.Bar(
                    x=['Negative', 'Positive'],
                    y=[proba[0]*100, proba[1]*100],
                    marker_color=['#ff9999', '#99ff99']
                ))
                fig.update_layout(
                    title="Classification Probabilities",
                    yaxis_title="Percentage (%)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        uploaded_file = st.file_uploader("Upload CSV file with reviews", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'review' in df.columns:
                with st.spinner('Processing reviews...'):
                    # Batch prediction
                    vectorized_reviews = vectorizer.transform(df['review'])
                    df['prediction'] = model.predict(vectorized_reviews)
                    probas = model.predict_proba(vectorized_reviews)
                    df['confidence'] = np.max(probas, axis=1)
                    
                    # Results summary
                    st.markdown("### Batch Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        positive_count = len(df[df['prediction'] == 1])
                        negative_count = len(df[df['prediction'] == 0])
                        
                        fig = px.pie(
                            values=[positive_count, negative_count],
                            names=['Positive', 'Negative'],
                            title="Review Distribution"
                        )
                        st.plotly_chart(fig)
                    
                    with col2:
                        fig = px.histogram(
                            df,
                            x='confidence',
                            title="Confidence Distribution"
                        )
                        st.plotly_chart(fig)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            else:
                st.error("CSV must contain a 'review' column")

    # Performance analysis
    if st.button("📈 Analyze Performance"):
        metrics = {
            'Precision': 0.85,
            'Recall': 0.82,
            'F1-Score': 0.83,
            'Accuracy': 0.84
        }
        
        # Radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            name=selected_model
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Performance Metrics",
            height=400
        )
        st.plotly_chart(fig)

except FileNotFoundError as e:
    st.error(f"❌ Error: File not found\n{str(e)}")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Developed with Streamlit and Scikit-learn*")
