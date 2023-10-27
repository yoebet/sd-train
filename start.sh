#!/bin/bash
gunicorn -w 3 --log-level debug -b 0.0.0.0:7003 "app:get()"

