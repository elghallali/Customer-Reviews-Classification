import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report

# Configuration
st.set_page_config(layout="wide", page_title="Customer Review Classification", page_icon="🖊")

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm_model.pkl"
}
VECTORIZER_PATH = BASE_DIR / "models/classical_m/tfidf_vectorizer.pkl"

try:
    # Load vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Interface
    st.title("🖊 Customer Review Classification")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        list(MODELS.keys())
    )

    # Load selected model
    model_path = BASE_DIR / f"models/classical_m/{MODELS[selected_model]}"
    model = joblib.load(model_path)

    # Input area
    example_review = st.text_area(
        "Enter a customer review",
        placeholder="Type your review here...",
        height=150
    )

    if example_review:
        # Prediction
        example_review_vectorized = vectorizer.transform([example_review])
        prediction = model.predict(example_review_vectorized)
        proba = model.predict_proba(example_review_vectorized)[0]

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            sentiment = "🟢 Positive" if prediction[0] == 1 else ("🔴 Negative" if prediction[0] == 0 else "🟡 Neutral")
            st.markdown(f"### Prediction: {sentiment}")
            st.markdown(f"### Confidence: {max(proba) * 100:.1f}%")

        with col2:
            # Confidence chart
            fig = go.Figure(go.Bar(
                x=['Negative', 'Neutral', 'Positive'],
                y=[proba[0]*100, proba[1]*100, proba[2]*100],
                marker_color=['#ff9999', '#ffcc66', '#99ff99']
            ))
            fig.update_layout(
                title="Classification Probabilities",
                yaxis_title="Percentage (%)"
            )
            st.plotly_chart(fig)

    # Metrics and additional analysis
    if st.button("📊 Analyze Performance"):
        # Dummy data for demonstration
        metrics = {
            'Precision': 0.85,
            'Recall': 0.82,
            'F1-Score': 0.83
        }

        # Radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Performance Metrics"
        )
        st.plotly_chart(fig)

        # Display classification report (dummy example)
        st.subheader("Detailed Classification Report")
        report = """ 
Precision: 0.85
Recall: 0.82
F1-Score: 0.83
Accuracy: 0.84
        """
        st.code(report, language='text')

    # Add data upload for bulk prediction
    st.sidebar.subheader("Bulk Prediction")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'review' in data.columns:
            predictions = model.predict(vectorizer.transform(data['review']))
            probabilities = model.predict_proba(vectorizer.transform(data['review']))
            data['prediction'] = predictions
            data['confidence'] = probabilities.max(axis=1) * 100
            st.subheader("Bulk Prediction Results")
            st.dataframe(data)
            st.download_button("Download Results", data.to_csv(index=False), "results.csv", "text/csv")
        else:
            st.sidebar.error("CSV file must contain a 'review' column.")

except FileNotFoundError as e:
    st.error(f"❌ Error: File not found\n{str(e)}")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Developer Student At ENIAD*")
