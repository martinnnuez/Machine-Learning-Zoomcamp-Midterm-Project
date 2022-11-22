# Imports 
import pickle

import pandas as pd
import numpy as np

import random
# Encoders 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.feature_extraction import DictVectorizer

# Metrics and Cross validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# Predictive model
from lightgbm import LGBMClassifier

import bentoml 

# Load data
datos = pd.read_csv('salary.csv')

# Data preparation
datos = datos.replace(" ?", np.nan)

# NA values imputation
datos["native-country"] = datos["native-country"].replace(np.nan," United-States")
datos["workclass"] = datos["workclass"].replace(np.nan," Private")

male = datos["sex"] == ' Male'
female = datos["sex"] == ' Female'
nans = datos['occupation'].isna()
na_male = np.logical_and(male, nans)
na_female = np.logical_and(female, nans)
length_male = sum(na_male)
length_female = sum(na_female)
replacement_male = random.choices([" Craft-repair"," Exec-managerial"," Prof-specialty"," Sales"], weights=[.25, .25,.25, .25], k=length_male)
replacement_female = random.choices([" Adm-clerical"," Other-service"," Prof-specialty"," Sales"], weights=[.25, .25,.25, .25], k=length_female)
datos.loc[na_male,'occupation'] = replacement_male
datos.loc[na_female,'occupation'] = replacement_female

# Filter hours lower than 0
datos = datos[datos["hours-per-week"] > 0]

# Create new variable
datos["capital-total"] = datos["capital-gain"] - datos["capital-loss"]  

# Factorize target variable
datos['salary-num'] = pd.factorize(datos.salary)[0]

# Preparing data for training
df_proc=datos.drop(["salary","education"], axis=1)

# Numerical scaler
numericas = ["age","fnlwgt","capital-gain","capital-loss","hours-per-week","capital-total"] 
scaler = StandardScaler()
scaler.fit(df_proc[numericas])
df_proc[numericas]=scaler.transform(df_proc[numericas])

# Categorical target encoders
encoder_NC = TargetEncoder()
encoder_NC.fit(df_proc['native-country'], df_proc['salary-num'])
df_proc['native-country'] = encoder_NC.fit_transform(df_proc['native-country'], df_proc['salary-num'])

encoder_W = TargetEncoder()
encoder_W.fit(df_proc['workclass'], df_proc['salary-num'])
df_proc['workclass'] = encoder_W.fit_transform(df_proc['workclass'], df_proc['salary-num'])

encoder_R = TargetEncoder()
encoder_R.fit(df_proc['race'], df_proc['salary-num'])
df_proc['race'] = encoder_R.fit_transform(df_proc['race'], df_proc['salary-num'])

# Data split into features and target
X = df_proc.loc[:, df_proc.columns != "salary-num"]
y = df_proc["salary-num"]

# Dict vectorizer for the rest of the categorical variables
# Numerical variables at this step
numerical = ["age","fnlwgt","capital-gain","capital-loss","hours-per-week","capital-total","education-num","race","native-country","workclass"]
# Categorical variables in which I want to perform DictVectorizer, so as to save the model and use it in the future. 
categorical = ["marital-status","occupation","relationship","sex"]
dicts = X[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)

# Train test spli
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Model fit
model = LGBMClassifier(
    learning_rate= 0.05,
    max_depth =-1, 
    min_child_samples= 5,
    n_estimators= 200,
    num_leaves=  35,
    reg_alpha =0.01)

model.fit(X_train, y_train)

# Save model with bentoml
bentoml.lightgbm.save_model(
    'salary_predict_model',
    model,
    custom_objects={
        'dictVectorizer': dv,
        "encoder_NC":encoder_NC,
        "encoder_W":encoder_W,
        "encoder_R":encoder_R,
        "scaler":scaler
    },
    signatures={
        'predict': {
            'batchable': True,
            'batch_dim': 0,
        }
    }
)




