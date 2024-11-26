# <div align="center">**Yelp Review Classification**</div>

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/Yelp_Logo.svg" alt="Yelp Logo" width="200">
</div>

---

## <div align="center">Project Overview</div>

This project aims to classify Yelp customer reviews into **sentiment categories** (e.g., positive/negative).  
It compares **classical machine learning algorithms** with **transfer learning models** (e.g., RoBERTa, XLNet) to determine the best approach for sentiment classification.

---

## <div align="center">Features</div>

- **Data Preprocessing**: Cleaning and preparing textual data.
- **Classical ML Models**: Implementing Logistic Regression and Random Forest.
- **Transfer Learning Models**: Fine-tuning RoBERTa and XLNet for text classification.
- **Evaluation**: Comparing models using metrics like accuracy, precision, recall, and F1-score.
- **Visualization**: Displaying results with confusion matrices and performance charts.

---

## <div align="center">Project Structure</div>

```plaintext
Yelp_Review_Classification/
├── data/
│   ├── raw/                # Raw Yelp dataset (subset)
│   ├── processed/          # Processed data (features and labels)
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_classical_ML_models.ipynb
│   ├── 3_transfer_learning_models.ipynb
│   ├── 4_evaluation_comparison.ipynb
├── models/
│   ├── classical_ml/       # Trained classical ML models
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   ├── transfer_learning/  # Fine-tuned transformer models
│       ├── roberta_model/
│       ├── xlnet_model/
├── reports/
│   ├── presentation.pptx   # Final presentation
│   ├── final_report.pdf    # Final report
├── requirements.txt        # Dependencies for the project
├── README.md               # Project documentation```
