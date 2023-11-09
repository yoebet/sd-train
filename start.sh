#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
source ../venv/bin/activate
gunicorn -w 3 --log-level debug -b 0.0.0.0:8005 "app:get()"

