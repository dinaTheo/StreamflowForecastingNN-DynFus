FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

