# House Prices Prediction using XGBoost

This project predicts house prices using the Kaggle **House Prices: Advanced Regression Techniques** dataset.

#Problem Statement
Build a regression model to predict house sale prices based on property features such as area, quality, neighborhood, and amenities.

#Data Cleaning & Feature Engineering
- Handled missing values using domain-specific logic
- Created indicator variables for Garage, Basement, and Fireplace
- Applied KNN imputation for `LotFrontage`
- Log-transformed target variable to reduce skewness

#Model
- Algorithm: XGBoost Regressor
- Target variable: `SalePrice` (log-transformed)
- Evaluation metric: RMSE

# Results
- Validation RMSE (log-scale):  0.13556645031558265
- Validation RMSE (original scale): 26748.292848384208

#How to Run
```bash
pip install -r requirements.txt
python xgboost_house_prices.py
```bash


# House Prices – Advanced Regression (Refactored & Leakage-Safe Version)

## Why I Am Re-Committing This Project

Advanced Regression for Home Prices (Refactored & Leakage-Safe Version) 

This is a completely revised version of my initial entry for the "House Prices – Advanced Regression Techniques" Kaggle competition.

After going over my earlier implementation, I found that there were improvements in:

* Preventing data leaks
* Consistency of pipelines
* Clarity of feature engineering
* Robust encoding

A more streamlined and production-ready machine learning workflow is reflected in this recommit.
This Version's Improvements 
##1. Strict Leakage Control
* The train and test datasets are always kept apart.
* Only training data is used to fit all encoders and transformations.
* Before modeling, a validation split is carried out.
* Only training-fitted objects are used to convert test data.
* Only training medians are used in neighborhood-based imputations.

## 2. Cleaner Encoding Strategy

* Binary features mapped consistently.
* Ordinal features encoded using explicit quality mappings.
* Nominal features encoded using OneHotEncoder(handle_unknown="ignore").
* Missing categorical values handled deterministically.

This removes manual column mismatch issues and improves stability.

### 3. Feature Engineering Enhancements

Added structural and interaction features:

* Qual_x_GrLivArea (interaction feature)
* Log transformations for skewed numerical variables
* Indicator features:

  * Garage_Indicator
  * Basement_Indicator
  * Pool_indicator
  * Fence_indicator

These features improve signal representation for tree-based models.

### 4. Proper Target Transformation

* Model trained on log1p(SalePrice)
* Predictions transformed back using expm1()
* RMSE evaluated in both log-space and original scale

This aligns with Kaggle’s evaluation metric.


### 5. Model and Validation

Model: XGBoost Regressor

Parameters:

* n_estimators = 1000
* learning_rate = 0.05
* max_depth = 5
* subsample = 0.8
* colsample_bytree = 0.8
* tree_method = "hist"

Validation RMSE (log-scale): 0.13956262496442182

Residual analysis and SHAP were used to interpret model behavior and identify weaknesses.

## What This Version Demonstrates

This refactor reflects improvement in:

* ML pipeline design
* Leakage prevention
* Encoding discipline
* Feature engineering strategy
* Model interpretability
* Clean validation practices

The focus was not only improving score, but improving engineering quality and reproducibility.


## Project Structure

train.csv
test.csv
submission.csv
notebook.ipynb
README.md-

## Dataset

Kaggle Competition: House Prices – Advanced Regression Techniques

# Future Improvements

* Full ColumnTransformer + Pipeline integration
* Cross-validation
* Hyperparameter tuning 


---

