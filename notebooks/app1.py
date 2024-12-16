import os
import joblib
import streamlit as st
from textblob import TextBlob
import numpy as np
import plotly.express as px
import pandas as pd

# Improved configuration for model paths
def get_project_root():
    """Find the root directory of the project."""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))

# Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)

# Colors for graphs
SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral': '#f1c40f',
    'negative': '#e74c3c'
}

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.history = []
        self.setup_models()

    def setup_models(self):
        """Load classification models and vectorizer with improved error handling."""
        try:
            # Dynamic paths for models
            project_root = get_project_root()
            
            # List of possible paths for model directories
            possible_model_dirs = [
                os.path.join(project_root, "models", "classical_ml"),
                os.path.join(project_root, "models", "classical_m"),
                os.path.join(project_root, "models"),
                os.path.join(os.path.dirname(project_root), "models")
            ]

            # Find the correct model directory
            models_dir = None
            for potential_dir in possible_model_dirs:
                if os.path.exists(potential_dir):
                    models_dir = potential_dir
                    break

            if not models_dir:
                st.error(f"No model directory found. Checked paths: {possible_model_dirs}")
                return

            # Paths for models
            vectorizer_paths = [
                os.path.join(models_dir, "tfidf_vectorizer.pkl"),
                os.path.join(models_dir, "vectorizer.pkl")
            ]
            lr_paths = [
                os.path.join(models_dir, "logistic_regression.pkl"),
                os.path.join(models_dir, "lr_model.pkl")
            ]
            svm_paths = [
                os.path.join(models_dir, "svm_model.pkl"),
                os.path.join(models_dir, "svm.pkl")
            ]

            # Function to load the first existing file
            def load_first_existing_file(file_paths):
                for path in file_paths:
                    if os.path.exists(path):
                        return joblib.load(path)
                return None

            # Load models
            self.vectorizer = load_first_existing_file(vectorizer_paths)
            self.models['logistic_regression'] = load_first_existing_file(lr_paths)
            self.models['svm'] = load_first_existing_file(svm_paths)

            # Check that all models are loaded
            if not all([self.vectorizer, 
                        self.models.get('logistic_regression'), 
                        self.models.get('svm')]):
                st.warning("Some models were not loaded correctly.")
                st.info(f"Files in {models_dir}: {os.listdir(models_dir)}")

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.exception(e)

    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob."""
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
                
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return "positive", [0.1, 0.2, 0.7]
            elif polarity < -0.1:
                return "negative", [0.7, 0.2, 0.1]
            else:
                return "neutral", [0.2, 0.6, 0.2]
                
        except Exception as e:
            st.error(f"TextBlob Error: {str(e)}")
            return "unknown", [0.0, 0.0, 0.0]

    def predict_sentiment(self, text, model_name):
        """Predict sentiment using a machine learning model."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            if not self.vectorizer:
                raise ValueError("Vectorizer not loaded")

            # Vectorize the text
            text_vectorized = self.vectorizer.transform([text])
            
            # Prediction
            model = self.models[model_name]
            prediction = model.predict(text_vectorized)
            probabilities = model.predict_proba(text_vectorized)[0]
            
            # Convert prediction to label
            sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_labels.get(prediction[0], "unknown")
            
            return sentiment, probabilities
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            return None, None

    def log_history(self, text, sentiment, probabilities):
        """Log the analysis in history."""
        if sentiment and probabilities is not None:
            self.history.append({
                'text': text,
                'sentiment': sentiment,
                'probabilities': probabilities
            })

def create_sentiment_distribution_chart(history):
    """Create a sentiment distribution chart."""
    if not history:
        return None
        
    sentiments = [entry['sentiment'] for entry in history if entry['sentiment'] != 'unknown']
    if not sentiments:
        return None
        
    fig = px.histogram(
        x=sentiments,
        title="Sentiment Distribution",
        color=sentiments,
        color_discrete_map=SENTIMENT_COLORS,
        labels={'x': 'Sentiment', 'count': 'Number of Analyses'}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Sentiment",
        yaxis_title="Number of Analyses"
    )
    
    return fig

def create_sentiment_probabilities_chart(history):
    """Create a chart for sentiment probabilities."""
    if not history:
        return None
    
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
    
    df = pd.DataFrame(proba_data)
    df_melted = df.melt(
        id_vars=['Sentiment'],
        var_name='Probability Type',
        value_name='Probability'
    )
    
    fig = px.bar(
        df_melted,
        x='Sentiment',
        y='Probability',
        color='Probability Type',
        title='Sentiment Probabilities by Category',
        color_discrete_map={
            'Negative': SENTIMENT_COLORS['negative'],
            'Neutral': SENTIMENT_COLORS['neutral'],
            'Positive': SENTIMENT_COLORS['positive']
        }
    )
    
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Probability"
    )
    
    return fig

def main():
    # Title and description
    st.title("🎭 Sentiment Analyzer - Customer Reviews")
    st.markdown("""
    This app uses machine learning models to analyze customer reviews.
    You can choose between different models and visualize the results.
    """)

    # Initialize the analyzer
    analyzer = SentimentAnalyzer()

    # User interface
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter your text here",
                height=150,
                placeholder="Type or paste your text here..."
            )
        
        with col2:
            model_choice = st.selectbox(
                "Choose a model",
                ["TextBlob", "Logistic Regression", "SVM"]
            )
            show_charts = st.checkbox("Display analysis charts", value=True)
            
            if st.button("Analyze", type="primary"):
                if not text_input:
                    st.warning("⚠️ Please enter text to analyze.")
                else:
                    with st.spinner("Analyzing..."):
                        # Analyze based on selected model
                        if model_choice == "TextBlob":
                            sentiment, probabilities = analyzer.analyze_with_textblob(text_input)
                        elif model_choice == "Logistic Regression":
                            sentiment, probabilities = analyzer.predict_sentiment(text_input, "logistic_regression")
                        else:  # SVM
                            sentiment, probabilities = analyzer.predict_sentiment(text_input, "svm")

                        # Display results
                        if sentiment and probabilities is not None:
                            # Log to history
                            analyzer.log_history(text_input, sentiment, probabilities)
                            
                            # Display results
                            sentiment_color = SENTIMENT_COLORS.get(sentiment, '#000000')
                            st.markdown(f"""
                            ### Analysis Result
                            - **Detected Sentiment:** <span style='color:{sentiment_color}'>{sentiment.upper()}</span>
                            - **Probabilities:**
                                - Negative: {probabilities[0]:.2%}
                                - Neutral: {probabilities[1]:.2%}
                                - Positive: {probabilities[2]:.2%}
                            """, unsafe_allow_html=True)

    # Display charts
    if show_charts and analyzer.history:
        st.markdown("### Results Visualization")
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
