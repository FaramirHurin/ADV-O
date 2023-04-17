FROM python:3.8.5-slim-buster

MAINTAINER Gian Marco Paldino <gpaldino@ulb.ac.be>

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install notebook 

