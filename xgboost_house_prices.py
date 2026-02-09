#!/usr/bin/env python
# coding: utf-8

# ### Import dataset

# In[82]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.width', 0)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ### Load Dataset

# In[83]:


df = pd.read_csv(r'C:\Users\MARIYA JOHN\Downloads\house-prices-advanced-regression-techniques\train.csv')
test_df = pd.read_csv(r'C:\Users\MARIYA JOHN\Downloads\house-prices-advanced-regression-techniques\test.csv')


# ### Data Cleaning

# In[84]:


df.info


# In[85]:


df.dtypes


# In[86]:


na = df[df['PoolQC'].isna() & df['Fence'].isna() & df['MiscFeature'].isna()]# starting with this ones and seeing how much of it have na in common
na.shape


# In[87]:


df = df.drop(columns = ['PoolQC','Fence','MiscFeature'],axis = 1)


# In[88]:


df.shape


# In[89]:


df = df.drop(columns = 'Alley')#Most of the values are NA


# In[90]:


na_values = df[df['GarageType'].isna() & df['GarageYrBlt'].isna() & df['GarageFinish'].isna() & df['GarageQual'].isna() & df['GarageCond'].isna() ]
na_values[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']]# since all of the above had similiar no of NA,checking


# In[91]:


na_values[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']].dtypes


# In[92]:


df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df['GarageFinish'] = df['GarageFinish'].fillna('None')
df['GarageQual'] = df['GarageQual'].fillna('None')
df['GarageCond'] = df['GarageCond'].fillna('None')
df['GarageType'] = df['GarageType'].fillna('No_Garage')
df['Garage_Indicator'] = df['GarageType'].apply(lambda x:0 if x =='No_Garage' else 1 )# Added a new feature to evalute if they have  a garage or not
df['Garage_Indicator'].value_counts()


# In[93]:


na_values2 = df[df['BsmtQual'].isna() & df['BsmtCond'].isna() & df['BsmtExposure'].isna() & df['BsmtFinType1'].isna() & df['BsmtFinType2'].isna() ]
na_values2[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]# Now proceeding to the other set where i found a similiarity in their number's


# In[94]:


na_values2[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].dtypes


# In[95]:


bsmt_categorical = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in bsmt_categorical:
    df[col] = df[col].fillna('None')


# In[96]:


df['Basement_Indicator'] = df['BsmtQual'].apply(lambda x: 0 if x=='None' else 1)
df['Basement_Indicator'].value_counts()


# In[97]:


df['FireplaceQu'] = df['FireplaceQu'].fillna('No_fireplace')


# In[98]:


df['fireplace_indicator'] = df['Fireplaces'].apply(lambda x:0 if x == 0 else 1 )


# In[99]:


df.isna().sum()


# In[100]:


df['LotFrontage'].value_counts()


# In[101]:


na_values3 = df[df['LotFrontage'].isna()]
na_values3


# In[102]:


Q1 = df['LotFrontage'].quantile(0.25)
Q3 = df['LotFrontage'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# In[103]:


df["LotFrontage"].describe()


# In[104]:


def get_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers
#get_outliers(df, "GrLivArea")
get_outliers(df, "LotFrontage")["LotFrontage"]


# In[105]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(df["LotFrontage"])


# In[106]:


plt.scatter(df["LotFrontage"], df['SalePrice'])


# In[107]:


df.loc[df["LotFrontage"] == 313, "LotFrontage"] = df["LotFrontage"].quantile(0.99)


# In[108]:


plt.scatter(df["LotFrontage"], df['SalePrice'])


# In[109]:


##found numeric values that are correlated
#Choose features most correlated with LotFrontage (e.g., 1stFlrSF, LotArea, GrLivArea).
#Only numeric features (or encode categorical features numerically).
corr = df.select_dtypes(include=['number']).corr()['LotFrontage'].sort_values(ascending=False)
print(corr.head(10))


# In[110]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
sns.boxplot(x ='Neighborhood',y='LotFrontage', hue = 'Street' ,data = df,width=.5)
plt.xticks(rotation=45)      # rotate labels for readability
plt.show()


# In[111]:


from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Select numeric features
features = ['LotFrontage','LotArea','1stFlrSF','GrLivArea']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# KNN imputer
imputer = KNNImputer(n_neighbors=5)
X_imputed_scaled = imputer.fit_transform(X_scaled)

# Inverse scale to get original units
X_imputed = scaler.inverse_transform(X_imputed_scaled)

# Put back in dataframe
df[features] = X_imputed


# In[112]:


df['Electrical'] = df['Electrical'].fillna('no_electricity')


# In[113]:


df.isna().sum()


# ### Feature Engineering

# In[114]:


#Now i have to encode categorical values and do skewing only necessary
#And FIND FEATURES FOR PREDICTING
#Put them as X nd Y


# In[115]:


y = df['SalePrice']
X = df.drop('SalePrice', axis=1)


# In[116]:


x_cat = X.select_dtypes(include='object').columns
x_cat

#Understanding whether it is Nominal, Ordinal or binary for encoding
f1_corrected = [
    'MSZoning: Nominal',
    'Street: Binary',
    'LotShape: Ordinal',
    'LandContour: Nominal',
    'Utilities: Binary',
    'LotConfig: Nominal',
    'LandSlope: Ordinal',
    'Neighborhood: Nominal',
    'Condition1: Nominal',
    'Condition2: Nominal',
    'BldgType: Nominal',
    'HouseStyle: Ordinal',
    'RoofStyle: Nominal',
    'RoofMatl: Nominal',
    'Exterior1st: Nominal',
    'Exterior2nd: Nominal',
    'MasVnrType: Nominal',
    'ExterQual: Ordinal',
    'ExterCond: Ordinal',
    'Foundation: Nominal',
    'BsmtQual: Ordinal',
    'BsmtCond: Ordinal',
    'BsmtExposure: Ordinal',
    'BsmtFinType1: Ordinal',
    'BsmtFinType2: Ordinal',
    'Heating: Nominal',
    'HeatingQC: Ordinal',
    'CentralAir: Binary',
    'Electrical: Nominal',
    'KitchenQual: Ordinal',
    'Functional: Ordinal',
    'FireplaceQu: Ordinal',
    'GarageType: Nominal',
    'GarageQual: Ordinal',
    'GarageCond: Ordinal',
    'PavedDrive: Binary',
    'SaleType: Nominal',
    'SaleCondition: Nominal'
]


# In[117]:


##Nominal encoding
t = df[['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
        'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
        'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 
        'GarageType', 'SaleType', 'SaleCondition']]

for col in t:
    print(f"--- {col} ---")
    print(t[col].value_counts())
    print("\n")


# In[118]:


# Define nominal columns
nominal_cols = ['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
                'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 
                'GarageType', 'SaleType', 'SaleCondition']
nominal_cols.append('HouseStyle')

# One-hot encode for nominal columns
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

# Checkiing the result
X.head


# In[119]:


#Encoding Binary columns
binary_cols = ['Street', 'Utilities', 'CentralAir', 'PavedDrive']

for col in binary_cols:
    # Convert column to categorical first
    X[col] = pd.Categorical(X[col]).codes


# In[120]:


#Ordinal Mapping
# Mapping for quality-type columns
quality_map = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1,
    'NoBasement': 0,   # for missing basements
    'No_fireplace': 0  # for missing fireplaces
}
 

# Dictionary of mappings for all columns
ordinal_mappings = {
    'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1},
    'LandSlope':{'Gtl': 3, 'Mod': 2, 'Sev': 1},
    'ExterQual': quality_map,
    'ExterCond': quality_map,
    'BsmtQual': quality_map,
    'BsmtCond': quality_map,
    'BsmtExposure': {'Gd':3,'Av':2,'Mn':1,'NoBasement':0,'No':0},
    'BsmtFinType1': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBasement':0},
    'BsmtFinType2': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NoBasement':0},
    'HeatingQC': quality_map,
    'KitchenQual': quality_map,
    'Functional': {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0},
    'FireplaceQu': quality_map,
    'GarageFinish': {'Fin':3,'RFn':2,'Unf':1,'NoGarage':0},
    'GarageQual': quality_map,
    'GarageCond': quality_map
}

# Apply mapping
for col, mapping in ordinal_mappings.items():
    X[col] = X[col].map(mapping)


# ### Preparation of test dataset

# In[121]:




# Save Id for kaggle submission
test_ids = test_df['Id']

test_df = pd.get_dummies(test_df, columns=nominal_cols, drop_first=True)
# Binary encoding
binary_cols = ['Street', 'Utilities', 'CentralAir', 'PavedDrive']

for col in binary_cols:
    # Convert column to categorical first
    test_df[col] = pd.Categorical(test_df[col]).codes


# Ordinal encoding

for col, mapping in ordinal_mappings.items():
    X[col] = X[col].map(mapping).fillna(0)
    test_df[col] = test_df[col].map(mapping).fillna(0)

#  Nominal one-hot encoding

existing_nominal_cols = [col for col in nominal_cols if col in X.columns]
test_df = pd.get_dummies(test_df, columns=existing_nominal_cols, drop_first=True)


# Align columns

missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0

test_df = test_df[X.columns]


# ### Training the data

# In[122]:


# Train / validation split

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Log-transform target

y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
# Train XGBoost on log-transformed target

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",   # good for larger datasets
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False
)


# ### Prediction & Validation Evaluation

# In[123]:


# Predict on validation set

y_val_pred_log = xgb_model.predict(X_val)   # log-space predictions
y_val_pred     = np.expm1(y_val_pred_log)   # back to original scale# Evaluate RMSE

rmse_log = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))
rmse     = np.sqrt(mean_squared_error(y_val, y_val_pred))

print("Validation RMSE (log-scale):", rmse_log)
print("Validation RMSE (original scale):", rmse)


# Predict on test set and save submission

test_pred_log = xgb_model.predict(test_df)  # predictions are already log1p(SalePrice)

test_pred = np.expm1(test_pred_log)  # expm1 undoes log1p



# ### Submission for Kaggle

# In[127]:


submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_pred
})
submission.to_csv(r'C:\Users\MARIYA JOHN\Downloads\submission.csv', index=False)


# In[ ]:




