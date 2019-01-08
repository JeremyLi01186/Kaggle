
# coding: utf-8

# In[1]:

# -*- Author Xieqin You, Xiaohe Yu, Yao Li for Machine Learning 6375 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cat
import warnings
warnings.simplefilter("ignore")
import gc
gc.enable()

# Flatten JSON
import os
import json
from pandas.io.json import json_normalize

columns_to_use = ['channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime', 'customDimensions']
JSON_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

def json_flatten(path):
    df = pd.read_csv(path, 
                     converters={column: json.loads for column in JSON_cols}, 
                     dtype={'fullVisitorId': str},
                     usecols=columns_to_use)
    # Normalize the JSON columns using json_normalize function
    for col in JSON_cols:
        json_df = json_normalize(df[col])
        json_df.columns = [f"{col}.{sub_col}" for sub_col in json_df.columns]
        df = df.drop(col, axis=1).merge(json_df, right_index=True, left_index=True)
    return df

# Load the train and test databases for V2 version
train = json_flatten("train_v2.csv")
test = json_flatten("test_v2.csv")

# Check the shape of the train and test databases
print(train.shape)
print(test.shape)

# Display the flattened columns
print(train.columns)

# Delete 'trafficSource.campaignCode' in train because test doesn't have such a column
train.drop('trafficSource.campaignCode', axis=1, inplace=True)

# Check how many unique items in each column
for col in train.columns:
    print(col, ": ", len(train[col].unique()))

# Drop those columns that have just one unique value 
for col in train.columns:
    if ((len(train[col].unique()) == 1) | (len(test[col].unique()) == 1)):
        train.drop(col, axis = 1, inplace = True)
        test.drop(col, axis = 1, inplace = True)

print(train.shape)
print(test.shape)

# Check the datatypes of all the columns
print(train.dtypes)

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Use the oldest date as the base data
# and count the days between each date and the base date
min_test_date = test['date'].min()
test.date = pd.to_datetime(test.date, format="%Y%m%d") - pd.to_datetime(min_test_date, format="%Y%m%d")
# Change timeDelta to integer 
test.date = test.date.dt.days

min_train_date = train['date'].min()
train.date = pd.to_datetime(train.date, format="%Y%m%d") - pd.to_datetime(min_train_date, format="%Y%m%d")
# Change timeDelta to integer 
train.date = train.date.dt.days

# Check the visitStartTime column
train.visitStartTime.head() 

# Drop the visitStartTime column because it contains too detailed information(record time in seconds)
train.drop('visitStartTime', axis=1, inplace=True)
test.drop('visitStartTime', axis=1, inplace=True)

# Drop the visitStartTime column because it contains too many unique values
# and we already have fullVisitorId information
train.drop('visitId', axis=1, inplace=True)
test.drop('visitId', axis=1, inplace=True)

print(train.shape)
print(test.shape)

# Get all categorical columns
categorical_cols = []

for col in train.columns:
    if train[col].dtype == 'object':
        categorical_cols.append(col)

categorical_cols.remove('fullVisitorId')

print(categorical_cols)


# In[2]:


# Remove rare values in each column and categorize such rare values as "others"
for col in categorical_cols:
    value_count = test[col].value_counts()
    frequent = value_count > 200
    common_vals = set(frequent.index[frequent].values)
    train.loc[train[col].map(lambda val: val not in common_vals), col] = 'others'
    test.loc[test[col].map(lambda val: val not in common_vals), col] = 'others'

# Check again if there's any column that has just one unique value
for col in train.columns:
    print(col, ": ", len(train[col].unique()))

# Delete columns that has just one unique value
for col in train.columns:
    if ((len(train[col].unique()) == 1) | (len(test[col].unique()) == 1)):
        train.drop(col, axis = 1, inplace = True)
        test.drop(col, axis = 1, inplace = True)

# Feature engineering by combine seperate columns to gether
for df in [train, test]:
    df['device'] = df['device.browser'] + '_' + df['device.deviceCategory'] + '_' + df['device.operatingSystem']
    df['geoNetwork'] = df['geoNetwork.city'] + '_' +df['geoNetwork.continent'] + '_' + df['geoNetwork.country'] +'_' + df['geoNetwork.metro'] + '_' +df['geoNetwork.networkDomain'] + '_' +df['geoNetwork.region'] + '_' +df['geoNetwork.subContinent']
    df['ad'] = df['trafficSource.adContent'].astype(str) +'_' +df['trafficSource.adwordsClickInfo.adNetworkType'].astype(str) + '_' + df['trafficSource.adwordsClickInfo.gclId'].astype(str) + '_' + df['trafficSource.adwordsClickInfo.slot'].astype(str)
    df['trafficSource'] = df['trafficSource.campaign'].astype(str) + '_' + df['trafficSource.isTrueDirect'].astype(str) + '_' + df['trafficSource.keyword'].astype(str)+ '_' + df['trafficSource.medium'].astype(str)+ '_' + df['trafficSource.referralPath'].astype(str) +'_' + df['trafficSource.source'].astype(str)
    
# Get the latest categorical columns and real columns
# Real columns represents original columns and catehorical_cols represent combined fields
categorical_cols = []
real_cols = []
for col in train.columns:
    if train[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        real_cols.append(col)

categorical_cols.remove('fullVisitorId')
#real_cols.remove('totals.transactionRevenue')

# Uncomment the following line if we'd like to drop the "totals.totalTransactionRevenue" column
#real_cols.remove('totals.totalTransactionRevenue')


# In[3]:


# Use LabelEncoder to encode categorical values
from sklearn.preprocessing import LabelEncoder

for column in categorical_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[column].values.astype('str')) + list(test[column].values.astype('str')))
    train[column] = lbl.transform(list(train[column].values.astype('str')))
    test[column] = lbl.transform(list(test[column].values.astype('str')))

for column in real_cols:
    train[column] = train[column].astype(float)
    test[column] = test[column].astype(float)

print(categorical_cols)
print(real_cols)

# Group all the rows by fullVisitorId 
for df in [train, test]:
    df= df.groupby("fullVisitorId")[categorical_cols + real_cols + ['totals.transactionRevenue']].mean().reset_index()
    
    
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold

# Use GroupKFold during validation
def validation(train, test, columns, model, n_splits, name="", fit_params={"early_stopping_rounds": 30, "verbose": 100, "eval_metric": "rmse", "categorical_feature":categorical_cols}):
    group_kfold = GroupKFold(n_splits) 
    test[name] = 0
    train[name] = np.NaN
    X_shuffled, y_shuffled, groups_shuffled = shuffle(train[columns], train["totals.transactionRevenue"], np.array(train['fullVisitorId']), random_state=1)
    for i, (train_idx, val_idx) in enumerate(group_kfold.split(X_shuffled, y_shuffled, groups_shuffled)):
        print("Fold ", i, ":")   
        model.fit(X_shuffled.iloc[train_idx], np.log1p(y_shuffled.iloc[train_idx]), eval_set=[(X_shuffled.iloc[val_idx], np.log1p(y_shuffled.iloc[val_idx]))], **fit_params)
        y_train_pred = model.predict(X_shuffled.iloc[val_idx])
        y_train_pred[y_train_pred < 0] = 0
        print("Fold ", i, " error: ", np.sqrt(mean_squared_error(np.log1p(y_shuffled.iloc[val_idx]), y_train_pred)))
        train[name][val_idx] = y_train_pred
        
        # Predict the log1p of revenue in test database    
        y_test_pred = model.predict(test[columns])
        y_test_pred[y_test_pred < 0] = 0
        # stack the test results
        test[name] += y_test_pred
    test[name] = test[name] / n_splits
      
# In[4]:
# Run LightGBM
lgbmodel = lgb.LGBMRegressor(n_estimators=1200, objective="regression", metric="rmse", learning_rate=0.01, subsample=.85)
validation(train, test, real_cols + categorical_cols, lgbmodel, 4, "lgb")


# Run CATBOOST
catmodel = cat.CatBoostRegressor(iterations = 500, learning_rate=0.03, depth=6, random_seed=0)
validation(train, test, real_cols + categorical_cols, catmodel, 4, "catboost", fit_params={"use_best_model": True, "verbose": 100})

# Stack LightGBM and CATBOOST together to reduce variances and wtite results to csv file
test['PredictedLogRevenue'] = 0.2 * test["lgb"] + 0.8 * test["catboost"]
test[['fullVisitorId','PredictedLogRevenue']].to_csv('submission_stack_new.csv', index = False)

