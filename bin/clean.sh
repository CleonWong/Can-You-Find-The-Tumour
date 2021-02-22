#!/bin/bash

rm -f logs/*.log
find . -name __pycache__ | xargs rm -rf
