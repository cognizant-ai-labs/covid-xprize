#!/usr/bin/bash

# run this script to initialize environment and install dependencies

rm -rf venv
mkdir venv
python3.7 -m venv venv
source ./venv/bin/activate
pip install -r  requirements.txt --upgrade
echo 'export PYTHONPATH="${PYTHONPATH}:$PWD"' >> ./venv/bin/activate