# Bank Customer Churn Prediction 

This is my final project for the [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main) course. The goal is to build an end-to-end machine learning project applying all the tools and concepts seen in the course. 
The following paragraphs will give an overview of the problem, the dataset, and the adopted methodology. You can skip to the instructions by clicking [here](#Instructions)

## Problem Description

The objective of this project is to predict whether a bank customer is likely to change provider or not. Churn is rather costly for any company, therefore there is an interest in knowing in advance which customers are more likely to leave, in order to be able to prevent the possible churn or actuate risk/loss mitigation procedures. Machine Learning models can help identifying potential quitters before they leave, allowing the bank to act on it in advance. 

## Dataset

The dataset used for this project contains information about bank customers, and can be found on [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn). For more information related to the dataset, you can consult the [data dictionary]([https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/data/README.md](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/data/data_dictionary.md ).

## Project Outline

### Setup
Conda and a pipfile environment have been created for this project. Requirements and dependences are all listed. Python 3.9.12 has been used. 
AWS has been used for storage and for running the MlFlow server. 
Docker and flask have been used for deployment. 

### Preliminary Analysis and Models Used 

An initial exploratory data analysis and first model experiments have been performed in two separate jupyter notebooks (stored in the [code](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/code/) folder), with the scope of defining a pre-processing method, selecting the models, and setting up the MLFlow toolkit. The ratio behind the choice of implemented Machine Learning models is explained in [MODELS.md](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/models/MODELS.md).

The models chosen to tackle the problem are Isolation Forests and XGBoost. The ratio behind this decision is explained in the [Model FAQ](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/models/MODELS.md).

### MLFlow

MLFow has been used for experiment tracking and model registry. Runs are saved in AWS S3, the PostgreSQL db collecting MlFlow data is hosted in AWS and the MLFlow server is run on an EC2 instance with elastic IP. 
AWS credentials are stored in an [.env](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/.env) file. The setup assumes that the final user has already configured the database, S3 bucket, and EC2 instance necessary for MLFlow to work. 
The MLFlow UI can be accessed at the following link:
````http:\\mlops-zoomcamp-mlflow.czwc40222kzm.eu-central-1.rds.amazonaws.com:5000````.


### Workflow Orchestration and Deployment
Prefect has been used for workflow orchestration, for model training and monitoring. 
The workflow for training models is fully deployed, with Process-type workers, and the code is stored in S3. The same S3 bucket used for MLFlow is employed also for prefect, but folders are diversified.
The prefect server is allocated on port 4200 and the UI can be accessed via [this link](http://127.0.0.1:4200) after starting the server from the terminal (````prefect server start````.).

### Deployment
The model is deployed locally and requests can be sent through port 9696. The app is wrapped in a a Docker container.

### Monitoring
Monitoring is done using Evidently AI. Data drift and model performances are tracked. Email alerts are sent in case of data drift or performance degradation, and, if data drift is detected, models are retrained and updated.
Evidently reports are stored in a [dedicated folder](https://github.com/FrancescaBellucci/mlops-zoomcamp/tree/main/final_project/monitoring/reports) and can be visualized in the notebook [_visualize_reports.jpynb_](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/code/visualize_reports.ipynb)).
The monitoring workflow can currently only be triggered manually by running ````python monitoring.py```` from the folder ````code````.

### Tests
*Unit tests are run on all the methods, excluding those in which files as saved, to avoid overwriting files that are crucial to the project. 
*There is an integration test. 

### Best Practices
* Linter and code formatters are used (isort, black, pylint)
* There is a Makefile
* There are pre-commit hooks

## Instructions





