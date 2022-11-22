import numpy as np

import pandas as pd

import bentoml

from bentoml.io import JSON

model_ref = bentoml.lightgbm.get("salary_predict_model:latest")
dv = model_ref.custom_objects["dictVectorizer"]
encoder_NC = model_ref.custom_objects["encoder_NC"]
encoder_W = model_ref.custom_objects["encoder_W"]
encoder_R = model_ref.custom_objects["encoder_R"]
scaler = model_ref.custom_objects["scaler"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("salary_predict_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(machine_data):
    
    query_df = pd.json_normalize(machine_data)

    # Preprocessing the data
    scale_var = ["age","fnlwgt","capital-gain","capital-loss","hours-per-week","capital-total"] 
    query_df[scale_var]=scaler.transform(query_df[scale_var])
    query_df["race"]=encoder_R.transform(query_df["race"])
    query_df["workclass"]=encoder_W.transform(query_df["workclass"])
    query_df["native-country"]=encoder_NC.transform(query_df["native-country"])

    # Dictvectorizer
    # Numerical variables at this step
    numerical = ["age","fnlwgt","capital-gain","capital-loss","hours-per-week","capital-total","education-num","race","native-country","workclass"]
    # Categorical variables in which I want to perform DictVectorizer, so as to save the model and use it in the future. 
    categorical = ["marital-status","occupation","relationship","sex"]

    query_dict = query_df[categorical + numerical].to_dict(orient='records')
    X_query = dv.transform(query_dict)
    

    y_pred = model_runner.predict.run(X_query)
    print(y_pred)
    
    if y_pred <= 0.5:
        return {"salary": "<=50K"}
    else:
        return {"salary": ">50K"}
