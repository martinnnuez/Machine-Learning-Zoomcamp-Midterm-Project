#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

query_id = 'Random person 1'
query = {
    "age":39,
    "workclass":" State-gov",
    "fnlwgt":77516,
    "education-num":13,
    "marital-status":" Never-married",
    "occupation":" Adm-clerical",
    "relationship":" Not-in-family",
    "race":" White",
    "sex":" Male",
    "capital-gain":2174,
    "capital-loss":0,
    "hours-per-week":40,
    "native-country":" United-States",
    "capital-total":2174
    }

response = requests.post(url, json=query).json()
print(response)

if response['salary_>50K'] == True:
    print('%s is rich' % query_id)
else:
    print('%s is not rich ' % query_id)