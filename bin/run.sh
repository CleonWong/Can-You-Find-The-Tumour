#!/bin/bash 

# --- First, run the program
python3 mammogram-cv.py

# --- Check to see if there was an error
#  make sure we redirect of the error message
logFile=$(ls logs/*.log 2> /dev/null | sort | tail -n1)

SHOW_ALL_LOG="no"
SHOW_TIME="no"

while getopts "at" option; do
    case "${option}" in
        a) SHOW_ALL_LOG="yes" ;;
        t) SHOW_TIME="yes"    ;;
    esac
done

# Turn this on if necessary
if [ "$SHOW_ALL_LOG" = 'yes' ]
then
    echo "The entire log file:"
    cat $logFile 
fi

# Find the timming information
if [ "$SHOW_TIME" = 'yes' ]
then
    echo "Timming information:"
    cat $logFile | grep 'seconds'
fi


# Print the errors in the log files ...
if [ -f "$logFile" ]; then
    # Print the errors
    echo "Errors:"
    cat $logFile | grep 'ERROR'
fi

exit 0 # Prevent an error call in the Makefile
