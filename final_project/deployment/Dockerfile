FROM python:3.9.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /web_service

#Copy Piplock and Pipfile.lock in the main folder
COPY ["./Pipfile", "./Pipfile.lock", "./"]

#Install pipenv from the deployment folder
RUN cd deployment && pipenv install --system --deploy

#Copy necessary files for prediction code to work in the correct folder
COPY ["./code/predict.py", "./code/model_training.py", "./code/__init__.py", "./code/"]
COPY ["./models/isolation_forest.bin", "./models/xgboost.bin", "./models/"]
COPY ["./data/customer_churn_training_data.parquet", "./data/"]

EXPOSE 9696

#Set working directory to be able to run code as written in predict.py
WORKDIR /web_service/code

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

#docker build -t bank-churn-prediction-service:v1 -f deployment/Dockerfile .
#docker run -it --rm -p 9696:9696 bank-churn-prediction-service:v1
#Manually add pipenv run pip install backports.tarfile
