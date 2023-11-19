#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi
nohup gunicorn -w 3 --log-level debug -b 0.0.0.0:8005 "app:get()" &

