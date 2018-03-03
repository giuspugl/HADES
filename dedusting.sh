#!/bin/bash
# run python code for input file number
source /opt/ioa/scripts/bashrc_linux
source /opt/ioa/setup/setup_modules.bash
module load anaconda2

# load directories
export FLIPPER_DIR=/home/ohep2/Masters/flipper/
export PATH=$PATH:$FLIPPER_DIR/bin
export PYTHONPATH=$PYTHONPATH:$FLIPPER_DIR/python

export FLIPPERPOL_DIR=/home/ohep2/Masters/flipperPol/
export PATH=$PATH:$FLIPPERPOL_DIR/bin
export PYTHONPATH=$PYTHONPATH:$FLIPPERPOL_DIR/python

# Hades Masters software
export HADES_DIR=/data/ohep2/hades
export PATH=$PATH:$HADES_DIR/bin
export PYTHONPATH=$PYTHONPATH:$HADES_DIR/python

# run code
python -u /data/ohep2/hades/dedusting.py $1
