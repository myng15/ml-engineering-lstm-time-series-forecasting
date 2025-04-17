# Start from python base image
FROM python:3.12.10-slim

# Change working directory
WORKDIR /code

# Copy requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Add python code
COPY ./app/ /code/app/

# Specify default commands
# Deploy to AWS ECS:
#CMD ["fastapi", "run", "app/run_inference.py", "--port", "80"]
# Deploy to AWS SageMaker:
CMD ["fastapi", "run", "app/run_inference_sagemaker.py", "--port", "8080"]



