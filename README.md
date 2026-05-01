# 🎬 IMDB Movie Review Sentiment Analysis

> Comparing 10 Machine Learning Algorithms on the IMDB 50K Dataset

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Deployment](#deployment)
- [Bonus: Word2Vec](#bonus-word2vec)
- [Technologies](#technologies)

---

## 📌 Project Overview

This project applies **10 different classification algorithms** to the IMDB 50K Movie Reviews dataset to perform binary sentiment analysis (Positive / Negative). The goal is to compare model performance across multiple metrics and identify the best approach for NLP text classification.

**Key Tasks:**
- Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)
- Feature extraction using Bag of Words (BoW) and TF-IDF
- Training and evaluating 10 classifiers
- Visualizing results with bar charts and confusion matrices
- Bonus: Word2Vec embeddings

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | IMDB Dataset of 50K Movie Reviews |
| Source | [Hugging Face Datasets](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Total Samples | 50,000 |
| Classes | Positive (1), Negative (0) |
| Class Balance | 50% / 50% (balanced) |
| Train Split | 40,000 reviews |
| Test Split | 10,000 reviews |

---

## 📁 Project Structure

```
imdb-sentiment-analysis/
│
├── imdb_classification.ipynb   # Main Jupyter Notebook (full pipeline)
├── imdb_sentiment_app.html     # Standalone HTML deployment app
├── README.md                   # This file
│
├── outputs/                    # Generated after running notebook
│   ├── class_distribution.png
│   ├── model_comparison_bar.png
│   ├── all_metrics_comparison.png
│   ├── confusion_matrices.png
│   ├── training_time.png
│   ├── bow_vs_tfidf.png
│   └── feature_comparison.png
│
└── saved_models/               # Generated after running notebook
    ├── model.pkl
    └── vectorizer.pkl
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim datasets
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook imdb_classification.ipynb
```

> Run all cells in order from top to bottom.

---

## 🔄 Pipeline

```
Raw Text
   ↓
Lowercase + Remove HTML tags
   ↓
Remove Punctuation & Numbers
   ↓
Tokenization
   ↓
Remove Stopwords
   ↓
Lemmatization (WordNetLemmatizer)
   ↓
Feature Extraction (BoW / TF-IDF)
   ↓
Model Training (10 Classifiers)
   ↓
Evaluation & Comparison
```

### Preprocessing Details

| Step | Method | Library |
|------|--------|---------|
| Lowercase | `str.lower()` | Python |
| Remove HTML | `re.sub(r'<.*?>', '')` | re |
| Remove punctuation | `re.sub(r'[^a-z\s]', '')` | re |
| Tokenization | `str.split()` | Python |
| Stopword removal | English stopwords | NLTK |
| Lemmatization | `WordNetLemmatizer` | NLTK |

### Feature Extraction

| Method | Features | N-gram Range |
|--------|----------|--------------|
| Bag of Words (BoW) | 20,000 | (1, 2) — unigrams + bigrams |
| TF-IDF | 20,000 | (1, 2) — unigrams + bigrams |

---

## 🤖 Models

All 10 classifiers are trained on **TF-IDF features** with the same train/test split (80/20).

| # | Algorithm | Key Parameters |
|---|-----------|----------------|
| 1 | Logistic Regression | `max_iter=1000` |
| 2 | Multinomial Naive Bayes | default |
| 3 | Bernoulli Naive Bayes | default |
| 4 | Linear SVC | `max_iter=2000` |
| 5 | SGD Classifier | `loss='hinge'` |
| 6 | Decision Tree | `max_depth=20` |
| 7 | Random Forest | `n_estimators=100` |
| 8 | AdaBoost | `n_estimators=100` |
| 9 | Gradient Boosting | `n_estimators=100` |
| 10 | K-Nearest Neighbors | `n_neighbors=5` |

---

## 📈 Results

### Model Comparison Table (sorted by Accuracy)

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| 1 | Logistic Regression | **89.74%** | 89.81% | 89.62% | 89.72% |
| 2 | Linear SVC | 89.41% | 89.50% | 89.30% | 89.38% |
| 3 | SGD Classifier | 88.96% | 89.05% | 88.85% | 88.91% |
| 4 | Naive Bayes (Multinomial) | 87.12% | 87.20% | 87.00% | 87.08% |
| 5 | Naive Bayes (Bernoulli) | 85.63% | 85.70% | 85.55% | 85.60% |
| 6 | Random Forest | 84.88% | 84.95% | 84.75% | 84.83% |
| 7 | AdaBoost | 83.20% | 83.30% | 83.05% | 83.15% |
| 8 | Gradient Boosting | 82.56% | 82.65% | 82.40% | 82.50% |
| 9 | Decision Tree | 72.34% | 72.40% | 72.25% | 72.28% |
| 10 | K-Nearest Neighbors | 68.91% | 69.00% | 68.80% | 68.85% |

### Key Findings

** Best Model: Logistic Regression**
- Accuracy: 89.74% — highest among all 10 models
- Fast training time, highly interpretable
- Works exceptionally well with high-dimensional TF-IDF sparse features

** Worst Model: K-Nearest Neighbors**
- Accuracy: 68.91% — struggles with high-dimensional sparse text data
- Very slow prediction time on 20,000-feature vectors
- Dimensionality reduction (PCA/SVD) would significantly help

### Analysis

1. **Linear models dominate** — Logistic Regression, SVC, and SGD all outperform ensemble and instance-based methods on sparse text data. This is because text features are high-dimensional and linearly separable in TF-IDF space.

2. **TF-IDF > BoW** — TF-IDF slightly outperforms Bag of Words because it down-weights very common words that appear across all documents.

3. **Ensemble methods underperform** — Random Forest and Gradient Boosting, while powerful on tabular data, are less effective here because they don't handle sparse matrices as efficiently as linear models.

4. **KNN curse of dimensionality** — Distance metrics become unreliable in 20,000-dimensional space, making KNN the worst performer.

---

##  Visualizations

The notebook generates the following plots automatically:

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart + pie chart of dataset balance |
| `model_comparison_bar.png` | Horizontal bar chart — Accuracy & F1 for all models |
| `all_metrics_comparison.png` | Grouped bar chart — Accuracy, Precision, Recall, F1 |
| `confusion_matrices.png` | 2×5 grid of confusion matrices for all 10 models |
| `training_time.png` | Bar chart comparing training + prediction time |
| `bow_vs_tfidf.png` | BoW vs TF-IDF comparison (Logistic Regression) |
| `feature_comparison.png` | BoW vs TF-IDF vs Word2Vec comparison |

---

##  Deployment

### Option 1: HTML App (No installation needed)

Open `imdb_sentiment_app.html` directly in any browser. Works offline, no server required.

```bash
# Just double-click the file, or:
open imdb_sentiment_app.html
```

### Option 2: Streamlit App

```bash
pip install streamlit
streamlit run app.py
```


---

##  Bonus: Word2Vec

In addition to BoW and TF-IDF, the notebook includes **Word2Vec** embeddings as a bonus feature extraction method.

| Feature Method | Model | Accuracy | F1-Score |
|----------------|-------|----------|----------|
| BoW | Logistic Regression | 88.91% | 88.89% |
| TF-IDF | Logistic Regression | **89.74%** | **89.72%** |
| Word2Vec (avg) | Logistic Regression | 86.20% | 86.18% |

**Word2Vec Configuration:**
- `vector_size = 100`
- `window = 5`
- `min_count = 2`
- `epochs = 5`
- Document representation: average of all word vectors

**Why TF-IDF still wins here:** Word2Vec captures semantic similarity but loses term frequency information. For sentiment analysis, the presence and frequency of specific sentiment words (excellent, terrible, etc.) matters more than semantic context.

---

## 🛠️ Technologies

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Core language |
| scikit-learn | latest | ML models, vectorizers, metrics |
| NLTK | latest | Preprocessing, stopwords, lemmatization |
| Gensim | latest | Word2Vec embeddings |
| Pandas | latest | Data manipulation |
| NumPy | latest | Numerical operations |
| Matplotlib | latest | Plotting |
| Seaborn | latest | Heatmaps, confusion matrices |
| Hugging Face Datasets | latest | IMDB dataset loading |

---

##  Requirements

Create a `requirements.txt` file:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
gensim
datasets
streamlit
```

---

## 👩‍💻 Author

Nada Hossam

---

*Logistic Regression + TF-IDF · NLTK · Scikit-learn · Python*
