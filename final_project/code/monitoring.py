import os
import sys
import shutil
import datetime
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from predict import fit_model, load_model, compute_predictions
from prefect import flow, task
from sendgrid import SendGridAPIClient
from evidently import ColumnMapping
from model_training import (
    TARGET,
    DATA_FILE,
    ISF_VARIABLES,
    train_model,
    prepare_dataset,
    preprocess_data,
    compute_isf_metrics,
)
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from prefect.task_runners import SequentialTaskRunner
from evidently.base_metric import Metric, InputData, MetricResult
from sendgrid.helpers.mail import Mail, HtmlContent
from evidently.model.widget import BaseWidgetInfo
from evidently.metric_preset import ClassificationPreset
from evidently.renderers.html_widgets import header_text, plotly_figure
from evidently.renderers.base_renderer import MetricRenderer, default_renderer

parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))

### Global variables

CURRENT_DATA_PATH = '../data/'
REFERENCE_DATA_PATH = '../monitoring/reference_data/'
REPORT_PATH = '../monitoring/reports/'

NUMERICAL_VARIABLES = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
    'Exited',
    'Complain',
    'Satisfaction Score',
    'Point Earned',
]
CATEGORICAL_VARIABLES = ['Geography', 'Gender', 'Card Type']

NOW = datetime.datetime.now()

COLUMN_MAPPING = ColumnMapping(
    target=TARGET,
    numerical_features=NUMERICAL_VARIABLES,
    categorical_features=CATEGORICAL_VARIABLES,
    prediction='Predictions',
)


### Custom metric and report for Isolatoin Forest
class MyMetricResult(MetricResult):
    pct_returning_cur: float
    pct_inliers_cur: float
    pct_returning_ref: float
    pct_inliers_ref: float


class MyMetric(Metric[MyMetricResult]):

    column_name: str

    def __init__(self, column_name: str, options: Optional[dict] = None):
        self.column_name = column_name
        super().__init__()

    def calculate(self, data: InputData) -> MyMetricResult:

        pct_returning_cur, pct_inliers_cur = compute_isf_metrics(
            data.current_data, data.current_data[self.column_name]
        )

        if data.reference_data is not None:
            pct_returning_ref, pct_inliers_ref = compute_isf_metrics(
                data.reference_data, data.reference_data[self.column_name]
            )
        else:
            pct_returning_ref, pct_inliers_ref = None, None

        return MyMetricResult(
            pct_returning_cur=pct_returning_cur,
            pct_inliers_cur=pct_inliers_cur,
            pct_returning_ref=pct_returning_ref,
            pct_inliers_ref=pct_inliers_ref,
        )


@default_renderer(wrap_type=MyMetric)
class MyMetricRenderer(MetricRenderer):

    def render_html(self, obj: MyMetric) -> List[BaseWidgetInfo]:

        metric_result = obj.get_result()

        figure = go.Figure(
            data=[
                go.Bar(
                    name='Current',
                    x=['pct_returning', 'pct_inliers'],
                    y=[metric_result.pct_returning_cur, metric_result.pct_inliers_cur],
                    marker_color='rgb(255,0,0)',
                ),  # Red for Current
                go.Bar(
                    name='Reference',
                    x=['pct_returning', 'pct_inliers'],
                    y=[metric_result.pct_returning_ref, metric_result.pct_inliers_ref],
                    marker_color='rgb(75,75,75)',
                ),  # Grey for Reference
            ]
        )

        figure.update_layout(
            barmode='group',
        )

        return [
            header_text(
                label=f"""The percentage of returning customers
                within inliers is: {metric_result.pct_inliers_cur}"""
            ),
            header_text(
                label=f"The percentage of inliers is: {metric_result.pct_inliers_cur}"
            ),
            plotly_figure(title="Metrics Comparison", figure=figure),
        ]


#####################
# Tasks and subflows


@flow
def split_data(current_data: pd.DataFrame, save_data: bool):
    current_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    preprocessed_data = preprocess_data(current_data)
    train_data, val_data = prepare_dataset(preprocessed_data, save_data)

    return train_data, val_data


@task
def load_datasets():
    print("Loading reference data...")

    reference_data = pd.read_parquet(DATA_FILE)
    reference_val_data = pd.read_parquet(
        REFERENCE_DATA_PATH + 'reference_validation_data.parquet'
    )

    print("Reference data loaded. Loading current data...")
    current_data_file = CURRENT_DATA_PATH + 'customer_churn_training_data.parquet'
    current_data = pd.read_parquet(current_data_file)

    # If the current dataset is more recent than the current training and validation data,
    # I split it into training and validation set
    if (
        os.stat(current_data_file).st_mtime
        > os.stat('../data/customer_churn_training_data.parquet').st_mtime
    ):
        current_train_data, current_val_data = split_data(current_data, save_data=False)

    else:
        current_train_data = pd.read_parquet(
            '../data/customer_churn_training_data.parquet'
        )
        current_val_data = pd.read_parquet(
            '../data/customer_churn_validation_data.parquet'
        )

    return {
        'reference_data': reference_data,
        'reference_val_data': reference_val_data,
        'current_data': current_data,
        'current_train_data': current_train_data,
        'current_val_data': current_val_data,
    }


@task
def compute_predictions_df(pred_data: pd.DataFrame, isf, xgb_classifier):

    pred_data['Anomalies'] = compute_predictions(isf, pred_data.drop(TARGET, axis=1))

    pred_data['Predictions'] = 1 - pred_data['Anomalies']
    idx_outliers = pred_data[pred_data['Predictions'] == 2].index

    X_pred = pred_data[pred_data['Predictions'] == 2].drop(
        [TARGET, 'Anomalies', 'Predictions'], axis=1
    )
    pred_xgb = compute_predictions(xgb_classifier, X_pred)

    pred_data.loc[idx_outliers, "Predictions"] = pred_xgb

    return pred_data


@task
def generate_report(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, report_name: str
):

    print("Creating and running report...")

    if report_name not in ['drift', 'isolation_forest', 'xgboost']:
        print(
            """Error: unknown report. report_name can only be
            \"drift\", \"isolation_forest\" or \"xgboost\"."""
        )
        return None

    if report_name == 'isolation_forest':
        report = Report(metrics=[MyMetric(column_name='Anomalies')], timestamp=NOW)

    elif report_name == 'xgboost':
        report = Report(metrics=[ClassificationPreset()], timestamp=NOW)

    else:
        report = Report(
            metrics=[
                ColumnDriftMetric(column_name='Age'),
                ColumnDriftMetric(column_name='Complain'),
                ColumnDriftMetric(column_name='NumOfProducts'),
                ColumnDriftMetric(column_name=TARGET),
                DatasetDriftMetric(),
            ],
            timestamp=NOW,
        )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=COLUMN_MAPPING,
    )

    print("Report created and run. Saving report...")

    report.save(REPORT_PATH + report_name + "_report.json")

    print("Report saved.")

    return report.as_dict()


### Main Flow
@flow(task_runner=SequentialTaskRunner())
def monitoring():
    print("Loading datasets...")

    datasets = load_datasets()
    reference_data = datasets['reference_data']
    reference_val_data = datasets['reference_val_data']
    current_data = datasets['current_data']
    current_train_data = datasets['current_train_data']
    current_val_data = datasets['current_val_data']

    print("Datasets loaded. Loading models...")

    isf = load_model("isolation_forest")
    xgb_classifier = load_model("xgboost")

    fit_model(isf, current_train_data)
    fit_model(xgb_classifier, current_train_data)

    print("Models loaded. Computing predictions...")

    current_pred_data = compute_predictions_df(current_val_data, isf, xgb_classifier)
    reference_pred_data = compute_predictions_df(
        reference_val_data, isf, xgb_classifier
    )

    print("Predictions computed. Generating reports...")
    isf_report = generate_report(
        reference_pred_data, current_pred_data, 'isolation_forest'
    )

    xgb_report = generate_report(reference_pred_data, current_pred_data, 'xgboost')

    drift_report = generate_report(reference_data, current_data, 'drift')

    print("Reports generated. Storing relevant results...")

    pct_returning_diff = (
        isf_report['metrics'][0]['result']['pct_returning_ref']
        - isf_report['metrics'][0]['result']['pct_returning_cur']
    )
    pct_inliers_diff = (
        isf_report['metrics'][0]['result']['pct_inliers_ref']
        - isf_report['metrics'][0]['result']['pct_inliers_cur']
    )

    recall_diff = (
        xgb_report['metrics'][0]['result']['reference']['recall']
        - xgb_report['metrics'][0]['result']['current']['recall']
    )
    f1_diff = (
        xgb_report['metrics'][0]['result']['reference']['f1']
        - xgb_report['metrics'][0]['result']['current']['f1']
    )

    isf_drift = [
        drift_report['metrics'][0]['result']['drift_detected'],
        drift_report['metrics'][1]['result']['drift_detected'],
        drift_report['metrics'][2]['result']['drift_detected'],
        drift_report['metrics'][3]['result']['drift_detected'],
    ]

    data_drift = drift_report['metrics'][4]['result']['dataset_drift']

    print("Results stored. Checking results and sending alerts...")

    ### Isolation Forest logic

    if pct_returning_diff > 0.05 or pct_inliers_diff > 0.1:

        print("Isolation Forest performance is degrading. Sending email alert...")

        html_content = f"""WARNING: Isolation Forest metrics lowered too much.\n
        The ratio of returning customers within anomalous data dropped by:
        {pct_returning_diff} and the ratio of inliers dropped by: {pct_inliers_diff} ."""

        send_email(html_content)

    ### XGBoost logic

    if recall_diff > 0.01 or f1_diff > 0.01:

        print("XGBoost performance is degrading. Sending email alert...")

        html_content = f"""WARNING: XGBoost metrics lowered too much. \n
        Recall dropped by: {recall_diff} and F1-score by: {f1_diff} ."""

        send_email(html_content)

    ### Drift Logic

    # Setting email alert for Isolation Forests
    if True in isf_drift:
        print(
            "Data drift detected in Isolation Forest variables. Sending email alert..."
        )

        html_content = f"""WARNING: A drift was detected in one or more of the employed
        to train the model Isolation Forest.
        You might want to investigate whether to change said variables. \n
        Drift detected in variables:
        {np.array(ISF_VARIABLES+[TARGET])[isf_drift].tolist()}"""

        send_email(html_content)

    # Retraining models
    if data_drift is True:

        print("Data drift detected. Sending email alert and retraining models...")

        html_content = (
            "WARNING:Data drift was detected. The models have been retrained."
        )

        send_email(html_content)

        print("Retraining isolation Forest...")
        train_model("isolation_forest", DATA_FILE, save_data=False)

        print("Retraining XGBoost classifier...")
        train_model("XGBoost", DATA_FILE, save_data=True)

        print("Models retrained.")

        # Updating reference data
        print("Updating reference data...")
        shutil.copy(DATA_FILE, REFERENCE_DATA_PATH + 'reference_data.parquet')
        shutil.copy(
            CURRENT_DATA_PATH + 'customer_churn_training_data.parquet',
            REFERENCE_DATA_PATH + 'reference_training_data.parquet',
        )
        shutil.copy(
            CURRENT_DATA_PATH + 'customer_churn_validation_data.parquet',
            REFERENCE_DATA_PATH + 'reference_validation_data.parquet',
        )


### Auxiliary functions


def send_email(html_content: str):

    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=RECIPIENT_EMAIL,
        subject='ALERT - Bank Churn Prediction',
        html_content=HtmlContent(html_content),
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))


if __name__ == "__main__":

    SENDER_EMAIL = 'your.sendgrid.email@domain.com'
    RECIPIENT_EMAIL = 'recipient.email@domain.com'
    SENDGRID_API_KEY = 'your Sendgrid API key'

    monitoring()
