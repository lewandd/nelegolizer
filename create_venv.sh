#!/bin/bash
# Create .nelegolizer-venv in $HOME/.local if doesn't exist

LOCAL_DIR=$HOME/.local
VENV_PATH=$LOCAL_DIR/.nelegolizer-venv

if [ ! -e $HOME/.local ];
then
    mkdir $HOME/.local &&
    echo "Created $HOME/.local directory"
fi

if [ ! -e $HOME/.local/nelegolizer-venv ];
then
    python3 -m venv $HOME/.local/nelegolizer-venv
fi

source $HOME/.local/nelegolizer-venv/bin/activate
python3 -m pip install -r requirements.txt --no-cache-dir
python3 -m pip install -e .

