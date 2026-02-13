#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import train_test_split
train_df = pd.read_csv(r'C:\Users\MARIYA JOHN\Downloads\house-prices-advanced-regression-techniques\train.csv')
test_df = pd.read_csv(r'C:\Users\MARIYA JOHN\Downloads\house-prices-advanced-regression-techniques\test.csv')
test_ids = test_df['Id']


# In[24]:


#Preprocessing of data, handled missing values, created indicator's and new inteaction feature Qual_x_GrLivArea
def preprocessing(df):
    df = df.copy()
    df['Fence_indicator'] = df['Fence'].notna().astype(int)
    df['Pool_indicator'] = df['PoolQC'].notna().astype(int)
    df = df.drop(columns = ['PoolQC','Fence','MiscFeature','Alley','HouseStyle'],axis = 1,errors='ignore')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['GarageFinish'] = df['GarageFinish'].fillna('None')
    df['GarageQual'] = df['GarageQual'].fillna('None')
    df['GarageCond'] = df['GarageCond'].fillna('None')
    df['GarageType'] = df['GarageType'].fillna('No_Garage')
    df['Garage_Indicator'] = df['GarageType'].apply(lambda x:0 if x =='No_Garage' else 1 )# Added a new feature to evalute if they have  a garage or not
    df['Garage_Indicator'].value_counts()
    bsmt_categorical = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in bsmt_categorical:
           df[col] = df[col].fillna('None')
    df['LotArea'] = np.log1p(df['LotArea'])
    df['FireplaceQu'] = df['FireplaceQu'].fillna('No_fireplace')
    df['Electrical'] = df['Electrical'].fillna('no_electricity')
    df['Qual_x_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_x_GrLivArea'] = np.log1p(df['Qual_x_GrLivArea'])
    df['GrLivArea'] = np.log1p(df['GrLivArea'])
    df['Basement_Indicator'] = df['BsmtQual'].apply(lambda x: 0 if x=='None' else 1)
    df['fireplace_indicator'] = df['Fireplaces'].apply(lambda x:0 if x == 0 else 1 )
    return df


# In[25]:


train_df = preprocessing(train_df)
test_df  = preprocessing(test_df)
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
#The Ames dataset creater has adviced to remove GrLivArea > 4000


# In[26]:


y = train_df['SalePrice']
x = train_df.drop('SalePrice', axis=1)
x_train,x_val,y_train,y_val = train_test_split( x, y, test_size=0.2, random_state=42)


# In[27]:


#Doing binary mapping
def binary_map(df):
    df = df.copy()
    b_map = {
        'Street': {'Grvl':0, 'Pave':1},
        'Utilities': {'NoSeWa':0, 'AllPub':1},
        'CentralAir': {'N':0, 'Y':1},
        'PavedDrive': {'N':0, 'P':1, 'Y':2}
    }
    for col, mapping in b_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    return df
x_train = binary_map(x_train)
x_val = binary_map(x_val)
test_df = binary_map(test_df)


# In[28]:


#Doing ordinal mapping
def ordinal_encoding(df):
    df = df.copy()
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoBasement': 0, 'No_fireplace': 0, 'None': 0}
    ordinal_mappings = {
        'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1},
        'LandSlope': {'Gtl': 3, 'Mod': 2, 'Sev': 1},
        'ExterCond': quality_map, 'BsmtQual': quality_map, 'BsmtCond': quality_map,
        'KitchenQual': quality_map, 'ExterQual': quality_map, 'HeatingQC': quality_map,
        'FireplaceQu': quality_map, 'GarageQual': quality_map, 'GarageCond': quality_map,
        'BsmtExposure': {'Gd':3,'Av':2,'Mn':1,'NoBasement':0,'No':0, 'None': 0},
        'BsmtFinType1': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBasement':0, 'None': 0},
        'BsmtFinType2': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBasement':0, 'None': 0},
        'Functional': {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0},
        'GarageFinish': {'Fin':3,'RFn':2,'Unf':1,'None':0}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    return df
x_train = ordinal_encoding(x_train)
x_val = ordinal_encoding(x_val)
test_df = ordinal_encoding(test_df)


# In[29]:


# 1. Calculate the medians per neighborhood using ONLY training data
#I selected Neighborhood over KNN imputer as it is more correlated to LotFrontage
neighborhood_medians = x_train.groupby('Neighborhood')['LotFrontage'].median()

# 2. Map those medians to the missing values in all datasets
x_train['LotFrontage'] = x_train['LotFrontage'].fillna(x_train['Neighborhood'].map(neighborhood_medians))
x_val['LotFrontage'] = x_val['LotFrontage'].fillna(x_val['Neighborhood'].map(neighborhood_medians))
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['Neighborhood'].map(neighborhood_medians))


# In[30]:


from sklearn.preprocessing import OneHotEncoder
nominal_cols = ['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
                'Condition1', 'Condition2', 'BldgType', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType',
                'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']
# Fill NaNs in categorical columns which was missed in MasVnrType
for col in nominal_cols:
    x_train[col] = x_train[col].fillna('None')
    x_val[col] = x_val[col].fillna('None')
    test_df[col] = test_df[col].fillna('None')

encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)

one_hot_encoded1 = encoder.fit_transform(x_train[nominal_cols])
x_train_half = pd.DataFrame(one_hot_encoded1, 
                           columns=encoder.get_feature_names(nominal_cols), 
                           index=x_train.index)
one_hot_encoded2 = encoder.transform(x_val[nominal_cols])
one_hot_encoded3 = encoder.transform(test_df[nominal_cols])
x_val_half = pd.DataFrame(one_hot_encoded2, 
                           columns=encoder.get_feature_names(nominal_cols), 
                           index=x_val.index)
test_df_half = pd.DataFrame(one_hot_encoded3, 
                           columns=encoder.get_feature_names(nominal_cols), 
                           index=test_df.index)
X_train_final = pd.concat([x_train.drop(columns=nominal_cols), x_train_half], axis=1)
X_val_final   = pd.concat([x_val.drop(columns=nominal_cols), x_val_half], axis=1)
test_df_final = pd.concat([test_df.drop(columns=nominal_cols), test_df_half], axis=1)


# In[31]:


import xgboost as xgb
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)


xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate= 0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",  
    random_state=42
)

xgb_model.fit(
    X_train_final,
    y_train_log,
    eval_set=[(X_val_final, y_val_log)],
    verbose=False
)


# In[32]:


from sklearn.metrics import mean_squared_error

y_val_pred_log = xgb_model.predict(X_val_final)   # log-space predictions
y_val_pred     = np.expm1(y_val_pred_log)   # back to original scale

rmse_log = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))
rmse     = np.sqrt(mean_squared_error(y_val, y_val_pred))# Evaluate RMSE

print("Validation RMSE (log-scale):", rmse_log)
print("Validation RMSE (original scale):", rmse)


test_pred_log = xgb_model.predict(test_df_final)  

test_pred = np.expm1(test_pred_log)


# In[33]:


submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_pred
})
submission.to_csv(r'C:\Users\MARIYA JOHN\Downloads\submission.csv', index=False)#Saving submission for Kaggle


# print(set(X_train_final.columns) - set(test_df_final.columns))
# print(set(test_df_final.columns) - set(X_train_final.columns))
# #checking if there are any data leakage

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(x=y_val_log, y=y_val_pred_log, ax=ax[0], alpha=0.5)
ax[0].plot([y_val_log.min(), y_val_log.max()], [y_val_log.min(), y_val_log.max()], '--r', lw=2)
ax[0].set_title('Predicted vs. Actual (Log Scale)')
ax[0].set_xlabel('Actual Log Price')
ax[0].set_ylabel('Predicted Log Price')
residuals_log = y_val_log - y_val_pred_log
sns.scatterplot( x = y_val_pred_log,y = residuals_log, ax=ax[1])
ax[1].axhline(y=0, color='r', linestyle='--')
ax[1].set_title('Residuals vs. Predicted Values')
ax[1].set_xlabel('Predicted Log Price')
ax[1].set_ylabel('Residual (Error)')


# In[35]:


#Because the Tree SHAP algorithm is implemented in XGBoost we can compute exact SHAP values quickly over thousands of samples.
#The SHAP values for a single prediction (including the expected output in the last column) sum to the model’s output for that prediction
import shap
explainer = shap.TreeExplainer(xgb_model,X_train_final)#(model, input)
shap_values = explainer(X_train_final)#input
shap.summary_plot(shap_values)


# The plot indicates that the highest feature "Qual_x_GrLivArea" pushes the model in prediction. 
# It was through my previous evaluation of shap values , I came to know that the highest feature that was contributing it wasn't pushing much. 
# so I introduced Qual_x_GrLivArea

# In[36]:


shap.plots.waterfall(shap_values[0])


# In[37]:


#A dependence scatter plot shows the effect a single feature has on the predictions made by the model.
#This means there are non-linear interaction effects in the model between Qual_x_GrLivArea  and other feature
shap.plots.scatter(shap_values[:,"Qual_x_GrLivArea"])


# In[38]:


#To show which feature may be driving these interaction effects we can color scatter plot by another feature. 
#If we pass the entire shap_values  to the color parameter then the scatter plot attempts to pick out the feature column with the strongest interaction with Qual_x_GrLivArea 
shap.plots.scatter(shap_values[:,"Qual_x_GrLivArea"], color= shap_values)


# This indicates that the a small curve that we had at the ending. The feature Qual_x_GrLivArea correaltes with TotalBSMTF.
# Even if two houses have the same "Quality x Living Area" score, the model predicts a higher price for the one with the bigger basement.

# Analysis of SHAP interaction values reveals that while Living Area is a primary driver, its impact is significantly magnified by the house quality and the presence of a large basement (TotalBsmtSF).
