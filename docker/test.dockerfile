# Build stage
FROM python:3.9-slim
# Create a working directory
WORKDIR /app

# Copy source code to working directory
COPY model/app.py /app/


# Copy app dependencies
RUN mkdir /app/src
COPY ./src /app/src

RUN mkdir /app/data

# make config folder and copy config.ini file
RUN mkdir /app/config
COPY ./config/config.ini /app/config


RUN pip install --upgrade cython
# Install the C compiler (gcc)
RUN apt-get update && apt-get install -y gcc

RUN apt-get install -y git

# Copy requirements.txt  file to working directory
COPY requirements.txt .

# Install packages from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN mkdir /app/tests
COPY ./tests /app/tests
