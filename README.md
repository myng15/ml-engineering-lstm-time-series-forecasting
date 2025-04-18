


# LSTM Time Series Forecasting with MLOps using MLflow & AWS Services

This project implements a pipeline for forecasting daily COVID-19 cases using LSTM networks. It supports model training, evaluation, visualization, MLflow tracking, and deployment through an inference API. MLOps best practices using tools like Docker, MLflow and AWS services (e.g. SageMaker, ECS, ECR, S3) are integrated to demonstrate a machine learning lifecycle from experimentation to production.


## Table of contents

* [Project Highlights](#project-highlights)
* [Getting Started](#getting-started)
* [Training](#training)
* [Tracking and Reproducing Experiments with MLflow](#tracking-and-reproducing-experiments-with-MLflow)
* [Deployment](#deployment)
* [TODOs](#todos)



## Project Highlights

- **Deep Learning Model:** Build and train an LSTM model for **time series forecasting** using PyTorch.
- **Experiment Tracking:** Utilize MLflow for tracking experiments, both locally and on an AWS EC2 instance.
- **Deployment:** Create a FastAPI application to serve forecasting predictions and deploy the application in a Docker container to AWS ECS (or host the trained model as an AWS SageMaker endpoint).


## Getting Started

### Prerequisites
- Python 3.12
- PyTorch 2.2.0
- MLflow 2.2+
- FastAPI
- Docker, Docker Hub
- AWS credentials and AWS CLI configured for ECR, SageMaker and S3 access
- AWS SDK (Boto3)


### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/myng15/ml-engineering-lstm-time-series-forecasting.git
   ```

2. **Set up the environment**:

   ```bash
   conda env create -f environment.yaml
   conda activate ts-lstm-mlflow-sagemaker
   ```

## Training

1. **Train the model**:

   ```bash
   python train.py 
   ```

You can also provide optional arguments:

    ```bash
    python train.py --test_size 0.25 --num_epochs 300 --learning_rate 0.001
    ```

2. **Finalize the model (i.e. train with all available data)**:

   ```bash
   python train.py --finalize_model
   ```

Here you can also provide additional arguments similar to `python train.py`.


## Tracking and Reproducing Experiments with MLflow

To enable logging hyperparameters, model performance and artifacts to MLflow and tracking them locally or remotely (e.g. on an EC2 instance), run:

  ```bash
   python mlflow/train_with_mlflow_tracking.py
  ```

To create a reproducible MLflow experiment, configure the hyperparameters and environment using the `MLproject` file, and then run, e.g.:

  ```bash
   mlflow run . --experiment-name EXP_NAME -P num_epochs=300 -P n_layers=3
  ```


## Running the Inference API

Once the model is finalized, you can serve (multi-day) forecast predictions via a FastAPI app defined in `app/run_inference.py`.

### **API Endpoints:**

`GET /`: Health check

`GET /predict?n_days=N`: Returns future N-day predictions based on query parameters.

> **For SageMaker integration:** Use `app/run_inference_sagemaker.py` instead:
>
> `GET /ping`: Health check
>
> `POST /invocations`: Accepts parameters as a JSON payload for serving predictions.


## Deployment

### AWS ECS

1. Push the Docker image of the FastAPI application to AWS ECR

2. Deploy to ECS: Use the AWS Management Console or AWS CLI to create a task definition and service using the pushed image

### AWS SageMaker

1. Create a SageMaker model

2. Deploy the model as SageMaker Endpoint

3. Invoke the endpoint to serve predictions


## TODOs

- [ ] Improve training strategy (e.g. cross-validation, ensemble of LSTMs)

- [ ] Quantify uncertainty in predictions

- [ ] Incorporate CI/CD for automated deployment

- [ ] Implement other forecasting models for comparison (e.g., XGBoost, GRU, Transformer, foundation models)

- [ ] Expand to other datasets, including multivariate time series problems


