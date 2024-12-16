import os
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.history = []
        self.setup_models()

    def setup_models(self):
        """Load classification models with improved error handling."""
        try:
            # Dynamic paths for models
            project_root = self._get_project_root()
            
            # Possible model directories
            possible_model_dirs = [
                os.path.join(project_root, "models", "classical_ml"),
                os.path.join(project_root, "models"),
                os.path.join(os.path.dirname(project_root), "models")
            ]

            # Find the correct model directory
            models_dir = next((path for path in possible_model_dirs if os.path.exists(path)), None)

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

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

    def _get_project_root(self):
        """Find the root directory of the project."""
        current_file = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(current_file))

    def analyze_sentiment(self, text, model_name='TextBlob'):
        """Analyze sentiment using different methods."""
        try:
            if model_name == 'TextBlob':
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    return "positive", [0.1, 0.2, 0.7]
                elif polarity < -0.1:
                    return "negative", [0.7, 0.2, 0.1]
                else:
                    return "neutral", [0.2, 0.6, 0.2]
            
            elif model_name in ['Logistic Regression', 'SVM']:
                if model_name == 'Logistic Regression':
                    model = self.models['logistic_regression']
                else:
                    model = self.models['svm']
                
                if not self.vectorizer or not model:
                    raise ValueError("Models not loaded correctly")
                
                text_vectorized = self.vectorizer.transform([text])
                prediction = model.predict(text_vectorized)
                probabilities = model.predict_proba(text_vectorized)[0]
                
                sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
                sentiment = sentiment_labels.get(prediction[0], "unknown")
                
                return sentiment, probabilities
            
            return "unknown", [0.3, 0.4, 0.3]
        
        except Exception as e:
            st.error(f"Sentiment Analysis Error: {str(e)}")
            return "unknown", [0.3, 0.4, 0.3]

    def analyze_dataframe(self, df, text_column, model_name='TextBlob'):
        """Analyze sentiment for entire DataFrame."""
        df['sentiment'], df['probabilities'] = zip(*df[text_column].apply(
            lambda x: self.analyze_sentiment(str(x), model_name)
        ))
        
        # Extract individual probabilities
        df['negative_prob'] = df['probabilities'].apply(lambda x: x[0])
        df['neutral_prob'] = df['probabilities'].apply(lambda x: x[1])
        df['positive_prob'] = df['probabilities'].apply(lambda x: x[2])
        
        return df

def create_3d_scatter_visualization(df):
    """Create 3D scatter plot of sentiment probabilities."""
    fig = go.Figure(data=[go.Scatter3d(
        x=df['negative_prob'],
        y=df['neutral_prob'],
        z=df['positive_prob'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['sentiment'].map({'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}),
            opacity=0.8
        ),
        text=df['sentiment']
    )])
    
    fig.update_layout(
        title='3D Sentiment Probability Distribution',
        scene=dict(
            xaxis_title='Negative Probability',
            yaxis_title='Neutral Probability',
            zaxis_title='Positive Probability'
        )
    )
    return fig

def create_sentiment_heatmap(df):
    """Create heatmap of sentiment probabilities."""
    sentiment_pivot = df.pivot_table(
        index='sentiment', 
        values=['negative_prob', 'neutral_prob', 'positive_prob'], 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(sentiment_pivot, annot=True, cmap='YlGnBu')
    plt.title('Average Sentiment Probabilities')
    return plt

def main():
    st.set_page_config(page_title="Advanced Sentiment Analyzer", layout="wide")
    
    st.title("🎭 Advanced Sentiment Analyzer")
    
    # Sidebar for navigation
    menu = ["Single Text Analysis", "Bulk CSV Analysis", "Advanced Visualization"]
    choice = st.sidebar.selectbox("Select Mode", menu)
    
    analyzer = EnhancedSentimentAnalyzer()
    
    if choice == "Single Text Analysis":
        # Single text analysis section (similar to previous implementation)
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
            
            if st.button("Analyze Text", type="primary"):
                if text_input:
                    sentiment, probabilities = analyzer.analyze_sentiment(text_input, model_choice)
                    
                    st.markdown(f"""
                    ### Analysis Result
                    - **Sentiment:** {sentiment.upper()}
                    - **Probabilities:**
                        - Negative: {probabilities[0]:.2%}
                        - Neutral: {probabilities[1]:.2%}
                        - Positive: {probabilities[2]:.2%}
                    """)
    
    elif choice == "Bulk CSV Analysis":
        st.header("Bulk Customer Review Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Select text column
            text_column = st.selectbox("Select text column for analysis", df.columns)
            model_choice = st.selectbox("Choose Analysis Model", ["TextBlob", "Logistic Regression", "SVM"])
            
            if st.button("Analyze CSV"):
                with st.spinner("Analyzing..."):
                    # Analyze entire DataFrame
                    analyzed_df = analyzer.analyze_dataframe(df, text_column, model_choice)
                    
                    # Display results
                    st.dataframe(analyzed_df)
                    
                    # Visualization options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(create_3d_scatter_visualization(analyzed_df))
                    
                    with col2:
                        st.pyplot(create_sentiment_heatmap(analyzed_df))
    
    elif choice == "Advanced Visualization":
        st.header("Advanced Sentiment Insights")
        
        uploaded_file = st.file_uploader("Upload CSV for Advanced Analysis", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Select text column
            text_column = st.selectbox("Select text column", df.columns)
            
            if st.button("Generate Advanced Visualizations"):
                with st.spinner("Processing..."):
                    # Analyze DataFrame
                    analyzed_df = analyzer.analyze_dataframe(df, text_column)
                    
                    # Create multiple visualization columns
                    cols = st.columns(3)
                    
                    # 3D Scatter Plot
                    with cols[0]:
                        st.plotly_chart(create_3d_scatter_visualization(analyzed_df))
                        st.markdown("**3D Probability Distribution**")
                    
                    # Sentiment Heatmap
                    with cols[1]:
                        st.pyplot(create_sentiment_heatmap(analyzed_df))
                        st.markdown("**Sentiment Probability Heatmap**")
                    
                    # PCA Visualization
                    with cols[2]:
                        # Prepare data for PCA
                        X = analyzed_df[['negative_prob', 'neutral_prob', 'positive_prob']]
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        pca = PCA(n_components=3)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        pca_df = pd.DataFrame(
                            data=X_pca, 
                            columns=['PC1', 'PC2', 'PC3']
                        )
                        pca_df['sentiment'] = analyzed_df['sentiment']
                        
                        fig = px.scatter_3d(
                            pca_df, 
                            x='PC1', 
                            y='PC2', 
                            z='PC3', 
                            color='sentiment',
                            title='PCA of Sentiment Probabilities'
                        )
                        st.plotly_chart(fig)
                        st.markdown("**PCA Sentiment Analysis**")

if __name__ == "__main__":
    main()
