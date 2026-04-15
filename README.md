# Quora Duplicate Questions Pair

A Streamlit demo for predicting whether two Quora-style questions are duplicates using handcrafted similarity features and an XGBoost model.

## Project Overview

This project includes:
- `app.py`: Streamlit app for interactive duplicate-question prediction.
- `utils.py`: preprocessing, feature engineering, vectorization, and prediction pipeline.
- `artifacts/`: saved model and preprocessing objects (`model.pkl`, `tfidf_vectorizer.pkl`, `scaler.pkl`).
- `notebooks/`: model training and exploration notebooks.

## How it Works

The app preprocesses question text, extracts custom features such as:
- question lengths and word counts
- shared token and stopword overlap
- substring similarity
- fuzzy string matching scores

It also vectorizes each question using a saved TF-IDF vectorizer, concatenates the features, scales the additional numeric features, and feeds the result into the saved XGBoost model.

## Installation

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install streamlit beautifulsoup4 fuzzywuzzy python-Levenshtein distance nltk scikit-learn scipy numpy pandas xgboost
```

3. Download or place the `artifacts/` folder in the project root.
4. If needed, download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## Running the App

From the project root:

```bash
streamlit run app.py
```

Then open the link shown in the terminal.

## Usage

- Enter two questions in the side-by-side text areas.
- Click `Predict duplicate`.
- The app displays whether the pair is predicted as `Duplicate` or `Not duplicate`, along with a confidence score.

## Notes

- The model expects the saved artifacts in `artifacts/`.
- The app relies on `utils.py` for preprocessing and feature construction.
- The feature pipeline was derived from the training notebook located in `notebooks/5_model_training.ipynb`.
