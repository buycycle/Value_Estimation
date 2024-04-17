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

# Copy requirements.txt file to working directory
COPY requirements.txt .

# Install packages from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt


WORKDIR /app

# Expose port 80
EXPOSE 80

ARG ENVIRONMENT
ENV ENVIRONMENT=${ENVIRONMENT}
ARG AB
ENV AB=${AB}

# Create a user to run the app
#ENV USER_ID=1001
#ENV GROUP_ID=1001

#RUN groupadd -g ${GROUP_ID} appgroup
#RUN useradd -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash appuser
#RUN chown -R appuser:appgroup .

#USER appuser


# uvicorn(fastapi) for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]


