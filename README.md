# Midterm project for the Machine Learning Zoomcamp

## Objectives of the project

* Think of a problem that's interesting for you and find a dataset for that
* Describe this problem and explain how a model could be used
* Prepare the data and doing EDA, analyze important features
* Train multiple models, tune their performance and select the best model
* Export the notebook into a script
* Put your model into a web service and deploy it locally with Docker
* Bonus points for deploying the service to the cloud

# Salary Prediction Classification 

Selected dataset: Salary Prediction Classification 

## Problem description

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

## Access the data
    
https://www.kaggle.com/datasets/ayessa/salary-prediction-classification/download?datasetVersionNumber=1

## Files

A) The first file is the [notebook](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/notebook.ipynb) where the data preparation, data cleaning, exploratory data analysis, model selection process and parameter tuning is done.

B) The [train](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/train.py) file is the file in charged of performing the data preparation, data cleaning and the model training of the best LGBMClassifier found when doing model optimization. It returns a pickle output file with the trained model and all the needed tools for performing preprocessing to the dataset.

C) The [predict](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/predict.py) file is the one that produces the flask web service in charged of receiving a query, preprocessing the data and retrieving a prediction. 

D) The [predict-test](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/predict-test.py) in charged of evaluating if the ```predict.py``` is working. 

E) The [bentomodel](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/bentomodel.ipynb) is the notebook in charged of performing the bentoml model process to train and save the model. 

F) The [trainbento](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/trainbento.py) in charged of training and retrieving a bentoml model. 

G) The [servicebento](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/servicebento.py) in charged of providing the service with the bentoml model. 

# Instructions to run:

## Complete and simple way:

* 1- Clone github repo:

```bash
git clone repo name
```

* 2 - Open a terminal and navigate to the directory.

* 3 - Run test.py to obtain the model and preprocessing tools.

```bash
pipenv run python train.py
```

If you prefer to install the dependencies locally, run:

```bash
pipenv install
```

* 4 - Run the predict.py app:

* a - Pipenv:

```bash
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

* b - Docker:

```bash
# Build
docker build -t appname .
 
# Connect
docker run -it --rm -p 9696:9696 appname
```

* 5 - Open a new terminal and test the app:

```bash
ipython predict-test.py

# OR

python predict-test.py
```

## Deployment locally with Bento

Run the [trainbento](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/trainbento.py) so as to save the best model as a bento model.

Then you can run it locally using the following command in the terminal

```bash
bentoml serve servicebento.py:svc
```
You can visit the [local host](http://0.0.0.0:3000/) to make predictions

Remember you need to pass a dictionary as follows:

```python
# Example query:
{"age":39,
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
 ```
## Another way is to deploy the model locally is to build the Bento, containerize and run it locally

After running the [trainbento](https://github.com/martinnnuez MachineLearningZoomcampMidtermProject/blob/main/trainbento.py):

1. In the project directory, run ```bentoml build```
2. Containerize the model by running ```containerize salary_predict_classifier:latest```
2. Run the docker container (replace {containerId} with the id of the container from the above command)```docker run -it --rm -p 3000:3000 salary_predict_classifier:latest serve --production```

# Deployment to the cloud

To perform the deployment to the cloud I choose Heroku. I will provide the complete process so as to deploy your model to the cloud using Heroku.

1. Run the [trainbento](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/trainbento.py) so as to save the best model as a bento model.

After the training and saving are done, you can run the below command to get a list of models in the BentoML store:

```bash
bentoml models list
```

You should be able to see the model you just trained. 

2. Create the service and verify it works. The service is loaded in the file [servicebento](https://github.com/martinnnuez/MachineLearningZoomcampMidtermProject/blob/main/servicebento.py)

To test it locally run: 

```bash
bentoml serve servicebento.py:svc
```

3. Build the Bento:

The term Bento means an archive that contains everything to run our services or API online, including all the code, models, dependency info, and configurations for setup.

Building it starts with creating a bentofile.yaml file in the directory [bentofile](https://github.com/martinnnuez MachineLearningZoomcampMidtermProject/blob/main/bentofile.yaml).

Once you have ir, build the bento.

```bash
bentoml build
```
4. Deploy to Heroku:

Call login to authenticate your terminal session:

```bash
heroku login
```
This opens up a tab in the browser where you can log in with your credentials. Next, log in to the container registry:

```bash
heroku container:login
```

Now, let's create an app named salarypreditclassifier:

```bash
heroku create salarypreditclassifier
```
Afterward, the app should be visible at https://dashboard.heroku.com/apps.

Now, we need to push our Bento to this app and deploy it online. To do that, we need to cd into the bento directory (which you can find with bentoml list) and inside the docker folder:

```bash
cd ~/bentoml/bentos/salary_predict_classifier/cz3rvgdg2oyrz27z/env/docker
```
From there, you call this command:

```bash
sudo DOCKER_BUILDKIT=1 heroku container:push web --app salarypreditclassifier --context-path=../..
```

```bash
heroku container:release web --app salarypreditclassifier
```

And finally you can access the app at:

https://salarypreditclassifier.herokuapp.com/

Check the app and try to use the model, with a query like this: 

```python
# Example query:
{"age":39,
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
 ```
 
### Conciderations:

1. Be sure to have previously downloaded the python image we are using, in our case is python:3.9.15-slim, you can do this by running:

```bash
docker run -it --rm  python:3.9.15-slim
```
2. If you have problems to login to heroku or the container or even with the deployment, run the command using sudo, that solved a lot of problems for me and was the solution. 
