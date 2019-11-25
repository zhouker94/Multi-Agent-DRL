FROM python:3.6.8-slim-jessie

LABEL maintainer="Owen Zhu" name="Multi-Agent DRL" version="0.1"

RUN pip install --no-cache --upgrade pip

COPY requirements.txt .

RUN find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf &&\
    pip install --no-cache -r requirements.txt

COPY src /src

WORKDIR /src
