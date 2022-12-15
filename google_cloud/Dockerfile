FROM python:3.8.6-buster

# copy files from local machine to image:
COPY moda /moda
COPY requirements.txt /requirements.txt
COPY authenticate-gcs.json /authenticate-gcs.json
COPY models /models



# specify commands to be run when image built
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# specify commands to be executed when container is run
CMD uvicorn moda.api.model_api:app --host 0.0.0.0
