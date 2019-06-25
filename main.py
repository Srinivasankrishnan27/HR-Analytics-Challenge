# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:12:07 2019

@author: Srinivasan T
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

log_transformer = FunctionTransformer(np.log1p, validate=True)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



def fill_na(df):
    df['department'].fillna('not provided', inplace=True)
    df['region'].fillna('not provided', inplace=True)
    df['education'].fillna('not provided', inplace=True)
    df['gender'].fillna('not provided', inplace=True)
    df['recruitment_channel'].fillna('not provided', inplace=True)
    df['recruitment_channel'].fillna('not provided', inplace=True)
    return df
    

train_data = pd.read_csv('./train_LZdllcl.csv',index_col='employee_id')
test_data = pd.read_csv('./test_2umaH9m.csv',index_col='employee_id')
cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']

# one hot encoding for following columns:
# department, region, education
# gender, recruitment_channel

train_data = fill_na(df=train_data)
test_data = fill_na(df=test_data)


train_data = pd.get_dummies(data=train_data,columns=cat_cols,drop_first=True)
train_data['previous_year_rating'].fillna(train_data['previous_year_rating'].mode()[0], inplace=True)

ref_cols = list(train_data.columns)
target = ref_cols.pop(ref_cols.index('is_promoted'))

test_data = pd.get_dummies(data=test_data,columns=cat_cols,drop_first=True)
test_data = test_data[ref_cols]



clf = RandomForestClassifier(n_estimators=100, random_state=2019,
                             class_weight='balanced')

clf.fit(X=train_data[ref_cols], y=train_data[target].values.tolist())

y_pred = clf.predict(train_data[ref_cols])
accuracy_score(y, y_pred)
