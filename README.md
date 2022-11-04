# Midterm project

This is my personal midterm project for the Machine Learning Zoomcamp.

## Objectives of the project

* Think of a problem that's interesting for you and find a dataset for that
* Describe this problem and explain how a model could be used
* Prepare the data and doing EDA, analyze important features
* Train multiple models, tune their performance and select the best model
* Export the notebook into a script
* Put your model into a web service and deploy it locally with Docker
* Bonus points for deploying the service to the cloud

## Selected dataset: Salary Prediction Classification 

This database consist of an extraction made by Barry Becker from the 1994 U.S. census database, and the prediction task is to determine whether a person makes over 50K a year. This problmes it is a binary classification problem where the objective class distribution is imbalanced. The dataset consists of 14 input variables where we can find categorical, ordinal, and numerical data. The complete list of variables is:

* Age.
* Workclass.
* Final Weight.
* Education.
* Education Number of Years.
* Marital-status.
* Occupation.
* Relationship.
* Race.
* Sex.
* Capital-gain.
* Capital-loss.
* Hours-per-week.
* Native-country.
* Income.

The objective of this problem is to develop an app that returns whereas a person earns more than 50 K or less. 

### Access the data
    
https://www.kaggle.com/datasets/ayessa/salary-prediction-classification/download?datasetVersionNumber=1

## Instructions to run:

* 1- Clone github repo:

```
git clone repo name
```

* 2 - Open a terminal and navigate to the directory.

* 3 - Run test.py to obtain the model and preprocessing tools.

```
pipenv run python train.py
```

If you prefer to install the dependencies locally, run:

```
pipenv install
```

* 4 - Run the predict.py app:

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

* 5 - Open a new terminal and test the app:

```
ipython predict-test.py

# OR

python predict-test.py
```