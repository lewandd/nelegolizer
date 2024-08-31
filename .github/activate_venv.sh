#!/bin/bash
# Activate $HOME/.local/nelegolizer-venv

LOCAL_DIR=$HOME/.local
VENV_PATH=$LOCAL_DIR/nelegolizer-venv

if [ ! -e $VENV_PATH ];
then
    echo "$VENV_PATH does not exist. Run .github/create_venv.sh."
else
    . $HOME/.local/nelegolizer-venv/bin/activate
fi
