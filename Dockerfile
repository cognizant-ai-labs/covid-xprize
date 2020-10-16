# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.
# DESCRIPTION: Docker file for covid-xprize
# Created to make CI with Codefresh easier

FROM python:3.6-slim

ENV APPS_HOME /usr/local/cognizant
ENV COVID_APP_HOME ${APPS_HOME}/covid-xprize

RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements file only. That way, we don't have to rebuild the requirements layer, which takes a long time,
# each time the source changes.
COPY ./requirements.txt ${COVID_APP_HOME}/requirements.txt
RUN pip3 install --no-cache-dir -r ${COVID_APP_HOME}/requirements.txt

# Copy rest of source dir
COPY ./ ${COVID_APP_HOME}

WORKDIR ${COVID_APP_HOME}

ENV PYTHONPATH=${COVID_APP_HOME}
