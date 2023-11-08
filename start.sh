#!/bin/bash
source ../venv/bin/activate
gunicorn -w 3 --log-level debug -b 0.0.0.0:8005 "app:get()"

