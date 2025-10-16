"""
Logistic Regression Model - Classifying Expensive Houses
--------------------------------------------------------
This script classifies whether a house is "expensive" or "not expensive"
based on the median price of the dataset using Logistic Regression.

Steps:
1. Load dataset
2. Create binary target variable (1 = price above median)
3. Select same features as in Linear Regression
4. Split data into train/test sets
5. Fit Logistic Regression model
6. Predict and evaluate using accuracy
7. Visualize confusion matrix and ROC curve
8. Discuss usefulness of Logistic Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# 1. Load dataset
data_path = "KC_housing_clean.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# 2. Create binary target variable
median_price = df["price"].median()
df["above_median"] = (df["price"] > median_price).astype(int)

print(f"Median house price: {median_price:,.2f}")
print(df["above_median"].value_counts())

# 3. Feature selection
"""
We’ll use the same subset of features as in the Linear Regression model
for consistency and comparability:
- sqft_living
- bedrooms
- bathrooms
- view
- condition
"""
features = ["sqft_living", "bedrooms", "bathrooms", "view", "condition"]
target = "above_median"

X = df[features]
y = df[target]

# 4. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# 5. Fit Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

print("\n Logistic Regression model trained successfully.")
print("Coefficients:", dict(zip(features, log_model.coef_[0])))

# 6. Predict and evaluate
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nClassification Accuracy: {accuracy:.3f}")

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Expensive", "Expensive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# 8. ROC Curve (Receiver Operating Characteristic)
y_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Expensive House Classification")
plt.legend()
plt.show()

# 9. Interpretation
"""
Interpretation:
---------------
Accuracy tells us how well the model distinguishes between expensive and
non-expensive houses. The confusion matrix helps identify whether the model
is biased toward one class.

ROC and AUC measure the model's ability to rank predictions correctly.
Higher AUC means better separability between classes.

Logistic Regression is useful when:
- The target variable is categorical (e.g., above/below median)
- We care about probabilities and classification thresholds
- We want an interpretable model that outputs odds ratios
"""
