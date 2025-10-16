"""
Comparison and Reflection - Linear vs Logistic Regression
---------------------------------------------------------
This script summarizes and compares the results and characteristics of the
Linear Regression and Logistic Regression models applied to the King County
Housing dataset.

Focus Areas:
1. Conceptual differences between models
2. Output types and evaluation metrics
3. Real-world use cases
4. Observations from this dataset
5. Optional: experimenting with feature inclusion
"""

import pandas as pd

# 1. Conceptual differences
"""
Linear Regression:
------------------
- Predicts a *continuous* numeric value (house price).
- The model learns a linear relationship between input features and price.
- Output: A numeric prediction, e.g. $550,000
- Evaluated using metrics like MSE (Mean Squared Error) or R².
- Sensitive to outliers; assumes a linear relationship between X and y.

Logistic Regression:
--------------------
- Predicts a *binary* outcome (expensive = 1, not expensive = 0).
- The model learns a linear boundary in log-odds space.
- Output: Probability between 0 and 1, e.g. 0.82 → “likely expensive”.
- Evaluated using metrics like Accuracy, Precision, Recall, ROC-AUC.
- Interpretable in terms of odds ratios (exp(β)).
"""

# 2. Summary of tasks and outputs
summary_data = {
    "Aspect": [
        "Prediction Type",
        "Example Output",
        "Target Variable",
        "Evaluation Metric",
        "Interpretation of Output",
        "Use Cases",
    ],
    "Linear Regression": [
        "Continuous",
        "Predicted Price = $545,230",
        "House Price ($)",
        "Mean Squared Error (MSE)",
        "Predicted price of the house",
        "Estimating market value or forecasting trends",
    ],
    "Logistic Regression": [
        "Categorical (Binary)",
        "Predicted Probability = 0.82 → Expensive",
        "Above/Below Median Price",
        "Accuracy, ROC-AUC, Confusion Matrix",
        "Probability that a house is expensive",
        "Classifying homes or leads for pricing strategy",
    ],
}

summary_df = pd.DataFrame(summary_data)
print("============== Model Comparison Summary ==============")
print(summary_df.to_string(index=False))

# 3. Observations from the dataset
"""
Observations:
-------------
- Linear regression produced an MSE value that reflects average squared deviation
  between actual and predicted prices. Depending on the scale of house prices,
  even small errors may appear large in absolute terms.
- Logistic regression provided a reasonably accurate classifier for whether a house
  is above the median price, with balanced accuracy between the two classes.
- Both models benefited from using key structural features (sqft_living, grade, etc.).
  Adding location-based features (like zip code or latitude/longitude) could further
  improve performance but would require encoding categorical data.


Reflection:
-----------
- Linear regression helps *estimate how much* a variable affects price.
- Logistic regression helps *decide if* a variable crosses a certain threshold (like affordability).
- In practice, real estate analytics often uses both:
    • Linear regression → for pricing predictions
    • Logistic regression → for risk or classification tasks
"""

# 4. Optional: future improvements
"""
Future Work:
------------
- Try PolynomialFeatures or Regularized Regression (Ridge/Lasso)
- Compare with Tree-based models (Random Forest, XGBoost)
- Use cross-validation for more robust evaluation
- Add feature scaling and encoding for categorical features
"""
