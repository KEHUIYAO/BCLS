#!/bin/bash

echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'GPU: ' `lspci | grep NVIDIA`

# Prepare the dataset
unzip data.zip

# Following the example from http://chtc.cs.wisc.edu/conda-installation.shtml
# except here we download the installer instead of transferring it
# Download a specific version of Miniconda instead of latest to improve
# reproducibility
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Update conda as workaround for https://github.com/conda/conda/issues/9681
# Will no longer be needed once conda >= 4.8.3 is available from repo.anaconda.com
conda install conda=4.8.3


# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f environment.yml

# Activate the environment and log all packages that were installed
conda activate pytorch-gpu

# install another package
conda install pytorch-lightning -c conda-forge

conda list

# Modify these lines to run your desired Python script
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
python main.py --dropout_rate=0.25 --training_data_size=200 --validation_data_size=20 --max_epoch=100

# create an output tar archive
tar -czf dropout_25.tar.gz tb_logs/Bayesian_ConvLSTM