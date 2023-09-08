#!/bin/bash
# Sep 8, 2023
# If you do not want to use .yml file and want to install things manually this bash file is offered to be run.

# creating a conda env
conda create -n Rocnet-env
conda activate Rocnet-env

# installing required  conda packages
conda install pytorch==1.12.1 -c pytorch
conda install easydict==1.9
conda install scipy  
conda install ipython
conda install jupyter
conda install matplotlib
conda install pip

# installing required pip packages
pip install -r requirements.txt

#jupyter notebook


