### Bank Customer Churn Prediction 

This is my final project for the MLOps ZoomCamp. The goal is to build an end-to-end machine learning project applying all the tools and concepts seen in the course. 

For each subfolder, you will find a README file describing its content and explaining the methodology and the logic behind the choices I made.

## Problem Description

The objective of this project is to predict whether a bank customer is likely to change provider or not. Churn is rather costly for any company, and Machine Learning can help identifying potential quitters before they leave, allowing the bank to act on it in advance. 

## Dataset

The dataset used for this project contains information about bank customers, and can be found on [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn). For more information related to the dataset, you can consult the [data dictionary](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/data/README.md).

## Project Outline

The models chosen to tackle the problem are Isolation Forests and XGBoost. The methodology adopted is the following: 
* An initial exploratory data analysis and first model experiments have been performed in jupyter notebook ([EDA]() and [first experiments]()), with the scope of defining a pre-processing method, selecting the models, and setting up the MLFlow toolkit.
* MLFow has been used for experiment tracking and model registry. Runs are saved locally and can be accessed via port 5000 after entering the following line in the terminal:
```` mlflow server --backend-store-uri sqlite:///bank_churn.db --default-artifact-root ../artifacts ````
* Prefect has been used for pipelines. The prefect server is allocated on port 2700 and the UI can be assessed via [this link]() after starting the server from the terminal.
* The model is deployed as a web service, hosted at port 9696 (link). The app is wrapped in a a Docker container. 


