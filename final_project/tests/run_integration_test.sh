#!/usr/bin/env bash

docker-compose up --build -d

sleep 5

pipenv run python integration.py

docker-compose down