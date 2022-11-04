import pickle

from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd


model_file = 'Optimized_LGBM'

with open(model_file, 'rb') as f_in:
    scaler, encoder_NC, encoder_W, encoder_R, dv, model = pickle.load(f_in)

app = Flask('salary')

@app.route('/predict', methods=['POST'])
def predict():
    query = request.get_json()
    query_df = pd.json_normalize(query)

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

    y_pred = model.predict_proba(X_query)[0, 1]
    if y_pred <= 0.5:
        salary = "<=50K"
    else:
        salary = ">50K"

    result = {
        'salary_>50K_probability': float(y_pred),
        'salary_>50K': bool(salary)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)