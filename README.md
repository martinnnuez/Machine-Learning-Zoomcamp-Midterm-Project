# Midterm project

    This is my personal midterm project to the Machine Learning Zoomcamp

## Objectives of the project

* Think of a problem that's interesting for you and find a dataset for that
* Describe this problem and explain how a model could be used
* Prepare the data and doing EDA, analyze important features
* Train multiple models, tune their performance and select the best model
* Export the notebook into a script
* Put your model into a web service and deploy it locally with Docker
* Bonus points for deploying the service to the cloud

## Dataset: Salary Prediction Classification 

    This database consist of an extraction made by Barry Becker from the 1994 U.S. census database, and the prediction task is to determine whether a person makes over 50K a year. This problmes it is a binary classification problem where the objective class distribution is imbalanced. The dataset consists of 14 input variables where we can find categorical, ordinal, and numerical data.

## Instructions to run:

* 1- Clone github repo:

```
git clone repo name
```

* 2 - Open a terminal and navigate to the directory.

* 3 - Run app:

* a - Pipenv:

```
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

* b - Docker:

```
# Build
docker build -t appname .
 
# Connect
docker run -it --rm -p 9696:9696 appname
```

* 4 - Open a new terminal and test the app:

```
ipython predict-test.py

# OR

python predict-test.py
```