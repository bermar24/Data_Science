"""
twitter_sentiment_main.py
Single script: EDA -> Preprocessing -> Feature creation -> Modeling -> Evaluation -> Plots
Usage: python twitter_sentiment_main.py --data_path data/twitter_sentiment.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.utils import class_weight

# For Word2Vec
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# For neural model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Plotting utilities
from wordcloud import WordCloud
import seaborn as sns

# -----------------------------
# 0. Arguments & Globals
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="twitter_data.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)
RND = args.random_state

# -----------------------------
# 1. Load dataset & quick EDA
# -----------------------------
df = pd.read_csv(args.data_path)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Basic checks
print("Missing values per column:\n", df.isna().sum())
print("Class distribution:\n", df['category'].value_counts())

# Quick distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='category', data=df, order=[-1,0,1])
plt.title("Sentiment distribution")
plt.savefig(os.path.join(args.output_dir, "sentiment_distribution.png"))
plt.show()

# -----------------------------
# 2. Preprocessing helpers
# -----------------------------
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)      # remove URLs
    text = re.sub(r'@\w+', '', text)         # remove mentions
    text = re.sub(r'[^a-z\s]', '', text)     # remove non-letter chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['clean_text'].astype(str).apply(clean_text)

# Add basic features helpful for EDA
df['num_chars'] = df['clean_text'].apply(len)
df['num_words'] = df['clean_text'].apply(lambda x: len(x.split()))
print(df[['num_chars','num_words']].describe())

# -----------------------------
# 3. Wordclouds (one per class)
# -----------------------------
def words_for_class(df, label):
    texts = df[df['category'] == label]['clean_text'].str.cat(sep=' ')
    return texts

for label in [-1, 0, 1]:
    wc = WordCloud(width=800, height=400, background_color='white').generate(words_for_class(df, label))
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for class {label}")
    fname = os.path.join(args.output_dir, f"wordcloud_{label}.png")
    plt.show()


# # -----------------------------
# # 4. Train/test split
# # -----------------------------
# X = df['clean_text'].values
# y = df['category'].values
# # map labels to 0,1,2 for convenience (or keep -1,0,1 but some sklearn expects non-negative)
# label_map = {-1:0, 0:1, 1:2}
# inv_label_map = {v:k for k,v in label_map.items()}
# y_mapped = np.array([label_map[i] for i in y])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=args.test_size, random_state=RND, stratify=y_mapped)
#
# # Optional: class weights for imbalanced dataset
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights_dict = {i: w for i,w in enumerate(class_weights)}
# print("Class weights:", class_weights_dict)
#
# # -----------------------------
# # 5. Feature: TF-IDF pipeline + classical models
# # -----------------------------
# tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
#
# # Example: Logistic Regression baseline with TF-IDF
# from sklearn.pipeline import make_pipeline
# lr_pipeline = make_pipeline(tfidf, LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RND))
#
# print("Training Logistic Regression (TF-IDF)...")
# lr_pipeline.fit(X_train, y_train)
# y_pred_lr = lr_pipeline.predict(X_test)
# print("LR Results:")
# print(classification_report(y_test, y_pred_lr, target_names=['neg','neu','pos']))
# print("Accuracy:", accuracy_score(y_test, y_pred_lr))
# print("F1 macro:", f1_score(y_test, y_pred_lr, average='macro'))
#
# # -----------------------------
# # 6. Feature: Word2Vec + average embeddings for classical models
# # -----------------------------
# # Train a small Word2Vec on training data tokens
# tokenized = [simple_preprocess(doc) for doc in X_train]
# w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4, seed=RND)
# w2v_model.save(os.path.join(args.output_dir, "word2vec.model"))
#
# def doc_vector(doc):
#     # average of token vectors for tokens present in w2v vocabulary
#     tokens = simple_preprocess(doc)
#     vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
#     if len(vecs) == 0:
#         return np.zeros(w2v_model.vector_size)
#     return np.mean(vecs, axis=0)
#
# # Create features
# X_train_w2v = np.vstack([doc_vector(d) for d in X_train])
# X_test_w2v  = np.vstack([doc_vector(d) for d in X_test])
#
# # Fit a Decision Tree or KNN on Word2Vec features
# dt_w2v = DecisionTreeClassifier(random_state=RND, class_weight='balanced')
# dt_w2v.fit(X_train_w2v, y_train)
# y_pred_dt_w2v = dt_w2v.predict(X_test_w2v)
# print("Decision Tree (Word2Vec) Results:")
# print(classification_report(y_test, y_pred_dt_w2v, target_names=['neg','neu','pos']))
#
# # -----------------------------
# # 7. Voting Classifier (TF-IDF features for LR, KNN, DT)
# # -----------------------------
# from sklearn.svm import LinearSVC
# knn = make_pipeline(tfidf, KNeighborsClassifier(n_neighbors=5))
# dt = make_pipeline(tfidf, DecisionTreeClassifier(random_state=RND))
# svc = make_pipeline(tfidf, LinearSVC(max_iter=5000))
# voting = VotingClassifier(estimators=[('lr', lr_pipeline), ('svc', svc), ('dt', dt)], voting='hard')  # hard voting
# print("Training Voting Classifier...")
# voting.fit(X_train, y_train)
# y_pred_voting = voting.predict(X_test)
# print("Voting Classifier Results:")
# print(classification_report(y_test, y_pred_voting, target_names=['neg','neu','pos']))
#
# # -----------------------------
# # 8. Neural model: CNN-LSTM hybrid
# # -----------------------------
# # Tokenization
# MAX_NUM_WORDS = 20000
# MAX_SEQ_LEN = 100
# EMBEDDING_DIM = 100
#
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
# tokenizer.fit_on_texts(X_train)
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq  = tokenizer.texts_to_sequences(X_test)
#
# X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
# X_test_pad  = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
#
# vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
# print("Vocab size:", vocab_size)
#
# def build_cnn_lstm_model(vocab_size, embedding_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN):
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
#     model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(LSTM(128, return_sequences=False))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(3, activation='softmax'))  # 3 classes
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# cnn_lstm = build_cnn_lstm_model(vocab_size)
# cnn_lstm.summary()
#
# # Callbacks
# es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# ckpt = ModelCheckpoint(os.path.join(args.output_dir, 'best_cnn_lstm.h5'), save_best_only=True, monitor='val_loss')
#
# print("Training CNN-LSTM model...")
# history = cnn_lstm.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[es,ckpt], class_weight=class_weights_dict)
#
# # Evaluate NN
# y_proba_nn = cnn_lstm.predict(X_test_pad)
# y_pred_nn = np.argmax(y_proba_nn, axis=1)
# print("CNN-LSTM Results:")
# print(classification_report(y_test, y_pred_nn, target_names=['neg','neu','pos']))
# print("Accuracy:", accuracy_score(y_test, y_pred_nn))
# print("F1 macro:", f1_score(y_test, y_pred_nn, average='macro'))
#
# # -----------------------------
# # 9. Save results table for comparison
# # -----------------------------
# results = []
# models_info = [
#     ('Logistic Regression (TF-IDF)', y_pred_lr),
#     ('Decision Tree (Word2Vec)', y_pred_dt_w2v),
#     ('Voting Classifier (TF-IDF)', y_pred_voting),
#     ('CNN-LSTM', y_pred_nn)
# ]
# for name, preds in models_info:
#     acc = accuracy_score(y_test, preds)
#     f1_macro = f1_score(y_test, preds, average='macro')
#     results.append({'model': name, 'accuracy': acc, 'f1_macro': f1_macro})
#
# results_df = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
# results_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
# print(results_df)
#
# # -----------------------------
# # 10. Confusion matrix plots
# # -----------------------------
# def plot_confusion(y_true, y_pred, title, fname):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6,5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg','neu','pos'], yticklabels=['neg','neu','pos'])
#     plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
#     # plt.savefig(fname)
#     plt.show()
#     plt.close()
#
# plot_confusion(y_test, y_pred_lr, "Confusion LR (TF-IDF)", os.path.join(args.output_dir, "cm_lr.png"))
# plot_confusion(y_test, y_pred_dt_w2v, "Confusion DT (W2V)", os.path.join(args.output_dir, "cm_dt_w2v.png"))
# plot_confusion(y_test, y_pred_voting, "Confusion Voting", os.path.join(args.output_dir, "cm_voting.png"))
# plot_confusion(y_test, y_pred_nn, "Confusion CNN-LSTM", os.path.join(args.output_dir, "cm_cnn_lstm.png"))
#
# # print("All outputs saved to", args.output_dir)
