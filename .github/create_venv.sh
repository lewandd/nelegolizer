#!/bin/bash
# Create nelegolizer-venv in $HOME/.local if doesn't exist

LOCAL_DIR=$HOME/.local
VENV_PATH=$LOCAL_DIR/nelegolizer-venv

if [ ! -e $LOCAL_DIR ];
then
    mkdir $LOCAL_DIR &&
    echo "Created $LOCAL_DIR directory"
fi

if [ ! -e $VENV_PATH ];
then
    python3 -m venv $VENV_PATH
    echo "Created $VENV_PATH venv"
fi

. $VENV_PATH/bin/activate
python3 -m pip install -r requirements.txt --no-cache-dir
python3 -m pip install -e .

