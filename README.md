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

