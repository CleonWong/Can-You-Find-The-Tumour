#!/bin/bash 

logFile=$(ls logs/*.log | sort | tail -n1)

cat $logFile
