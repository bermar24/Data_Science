# Twitter Sentiment Analysis Using Classical & Neural Models

## 1. Introduction
This project studies sentiment classification on Twitter using the Kaggle "Twitter Sentiment Dataset" (Saurabh Shahane, 2021). The dataset contains cleaned tweets (`clean_text`) and sentiment labels in `category` with values -1 (negative), 0 (neutral), +1 (positive). The goal is to compare classical machine learning models against a text-oriented neural model (CNN–LSTM hybrid) and identify which approach is best for multiclass sentiment classification in terms of accuracy and robust F1 (macro).


## 🎯 Research Questions Addressed
1. Which model achieves the best overall and per-class performance for predicting sentiment?
2. How do TF-IDF and Word2Vec features compare when used with classical models?
3. Does a CNN–LSTM hybrid outperform classical ensembles (Voting Classifier) on this dataset?

## 3. Dataset
- Source: Kaggle — Twitter Sentiment Dataset (Saurabh Shahane, 2021).
- Columns: `clean_text` (string), `category` (int: -1, 0, 1).
- Size: (162980, 2)

## 📁 Project Structure

- `twitter_sentiment_main.py` - Main pipeline script that implements the entire analysis
- `Twitter_Data.csv` - Dataset with cleaned tweets and sentiment labels
- `Twitter_Sentiment_Analysis.md` - Project specification document
- `requirements.txt` - Python dependencies

## 4. Methodology
1. EDA & preprocessing: class distribution, text length, missing data handling, tokenization, stopword removal.
2. Feature pipelines:
    - TF-IDF vectorizer (for baseline classical models).
    - Word2Vec embeddings (gensim): average tweet vectors for classical models.
    - Keras Tokenizer + padding for NN (embedding layer + CNN–LSTM).
3. Models:
    - Baselines: Decision Tree, KNN, Logistic Regression.
    - Ensemble: Voting Classifier (hard or soft) combining best performing classical models.
    - Neural: CNN–LSTM hybrid (Embedding → Conv1D → MaxPool → LSTM → Dense).
4. Hyperparameter tuning: GridSearchCV for classical models; manual / Keras Tuner for NN if time permits.
5. Evaluation: accuracy, precision, recall, F1 (macro & per class), confusion matrix, ROC-AUC (one-vs-rest).


## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline

**Basic usage (assuming Twitter_Data.csv is in the same directory):**
```bash
python twitter_sentiment_main.py
```

**With custom data path:**
```bash
python twitter_sentiment_main.py --data_path /path/to/Twitter_Data.csv
```

**With custom random state for reproducibility:**
```bash
python twitter_sentiment_main.py --random_state 123
```

## 📊 What the Pipeline Does

1. **Exploratory Data Analysis (EDA)**
    - Analyzes sentiment distribution
    - Examines text length statistics
    - Creates visualizations

2. **Data Preprocessing**
    - Cleans text (removes URLs, mentions, hashtags)
    - Removes stopwords
    - Tokenizes text
    - Sample the dataset into a more manageable size for testing (optional)

3. **Feature Engineering**
    - **TF-IDF Features**: Creates term frequency-inverse document frequency vectors
    - **Word2Vec Features**: Generates word embeddings and averages them per tweet
    - **Neural Network Features**: Uses Keras tokenizer and padding for sequence data

4. **Model Training**
    - **Classical Models**: Decision Tree, KNN, Logistic Regression (with GridSearchCV)
    - **Ensemble**: Voting Classifier combining best models
    - **Neural Network**: CNN-LSTM hybrid architecture

5. **Evaluation**
    - Accuracy, Precision, Recall, F1-Score (macro and per-class)
    - Confusion matrices
    - ROC curves
    - Model comparison table

## 📈 Output Files

After running the pipeline, you'll get:

- **Data Files**:
    - `model_comparison.csv` - Table comparing all models
    - `best_model.h5` - Best neural network weights

## 📝 Notes

- The script automatically downloads required NLTK data (punkt tokenizer and stopwords)
- Training time varies based on dataset size and hardware (expect 10-30 minutes for full pipeline)
- Neural network training uses early stopping to prevent overfitting
- All models are saved for later use/deployment

## 🔧 Troubleshooting

If you encounter memory issues with large datasets:
- Reduce `max_features` in TF-IDF vectorizer
- Decrease `batch_size` in neural network training
- Use a smaller subset of data for initial testing

## 📚 Citation

Dataset source: HUSSEIN, SHERIF (2021), "Twitter Sentiments Dataset", Mendeley Data, V1, doi:10.17632/z9zw7nt5h2.1
