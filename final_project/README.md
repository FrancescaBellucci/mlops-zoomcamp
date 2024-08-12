# Bank Customer Churn Prediction 

This is my final project for the [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main) course. The goal is to build an end-to-end machine learning project applying all the tools and concepts seen in the course. 
For each subfolder, you will find a README file describing its content and explaining the methodology and the logic behind the choices I made.
The following paragraphs will give an overview of the problem, the dataset, and the adopted methodology. You can skip to the instructions by clicking [here](#Instructions)

## Problem Description

The objective of this project is to predict whether a bank customer is likely to change provider or not. Churn is rather costly for any company, therefore there is an interest in knowing in advance which customers are more likely to leave, in order to be able to prevent the possible churn or actuate risk/loss mitigation procedures. Machine Learning models can help identifying potential quitters before they leave, allowing the bank to act on it in advance. 

## Dataset

The dataset used for this project contains information about bank customers, and can be found on [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn). For more information related to the dataset, you can consult the [data dictionary]([https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/data/README.md](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/data/data_dictionary.md ).

## Project Outline

The models chosen to tackle the problem are Isolation Forests and XGBoost. The methodology adopted is the following: 
* An initial exploratory data analysis and first model experiments have been performed in two separate jupyter notebooks (stored in the [code](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/code/) folder), with the scope of defining a pre-processing method, selecting the models, and setting up the MLFlow toolkit. The ratio behind the choice of implemented Machine Learning models is explained in [MODELS.md](https://github.com/FrancescaBellucci/mlops-zoomcamp/blob/main/final_project/models/MODELS.md).
* MLFow has been used for experiment tracking and model registry. Runs are saved locally and can be accessed via port 5000 after entering the following line in the terminal:
```` mlflow server --backend-store-uri sqlite:///bank_churn.db --default-artifact-root ../artifacts ````
* Prefect has been used for pipelines. The prefect server is allocated on port 4200 and the UI can be accessed via [this link](http://127.0.0.1:4200) after starting the server from the terminal.
* The model is deployed locally and requests can be sent through port 9696. The app is wrapped in a a Docker container.
* Monitoring is done using Evidently AI. Data drift and model performances are tracked. Email alerts are sent in case of data drift or performance degradation, and, if data drift is detected, models are retrained and updated.  

## Instructions




