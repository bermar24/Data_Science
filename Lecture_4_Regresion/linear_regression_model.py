"""
Linear Regression Model - Predicting House Prices
--------------------------------------------------
This script loads the cleaned King County Housing dataset and fits a
Linear Regression model to predict house prices.

Steps:
1. Load and inspect dataset
2. Select relevant features
3. Visualize feature relationships
4. Split data into train/test sets
5. Fit Linear Regression model
6. Predict and evaluate performance using MSE
7. Plot true vs predicted prices
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load dataset
data_path = "KC_housing_clean.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

# 2. Feature selection
"""
We’ll select a subset of features that are known to have a strong impact on price:
- sqft_living: main living area size (highly correlated with price)
- bedrooms: number of bedrooms
- bathrooms: number of bathrooms
- view: quality of the view
- condition: overall construction quality


These features are interpretable and avoid multicollinearity issues.
"""
features = ["sqft_living", "bedrooms", "bathrooms", "view", "condition"]
target = "price"

X = df[features]
# X = df.drop(columns=[target]) # Use all features except target
# X = pd.get_dummies(X, drop_first=True)
y = df[target]

# 3. Visualize feature relationships
sns.pairplot(df, x_vars=features, y_vars="price", height=3, aspect=1, kind="scatter")
plt.suptitle("Feature vs Price Relationships", y=1.02)
plt.show()

# 4. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# 5. Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n Linear Regression model trained successfully.")
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(features, model.coef_)))

# 6. Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")

# 7. Plot true vs predicted prices
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("True vs Predicted House Prices")
plt.show()

# 8. Interpretation
"""
MSE Interpretation:
-------------------
The Mean Squared Error represents the average squared difference between
actual and predicted prices. A smaller MSE indicates better predictive accuracy.
However, since house prices are large numbers, even a few thousand in MSE
can be reasonable. Comparing this to the price scale helps gauge model quality.
"""
