#!/bin/bash
#
# Test PyPi package
#


TESTS_PATH=$1
PYTORCH_VERSION=$2

if [[ -z $TESTS_PATH ]]; then
    echo "Usage: $0 TESTS_PATH [PYTORCH_VERSION]"
    exit 1
fi

if [[ -z $PYTORCH_VERSION ]]; then
  PYTORCH="pytorch"  # Use latest
else
  PYTORCH="pytorch==$PYTORCH_VERSION"  # Use latest
fi

source ~/Documents/miniconda3/etc/profile.d/conda.sh

ENV_NAME=torchac_test
conda env list | grep $ENV_NAME
if [[ $? == 1 ]]; then  # Env does not exist yet.
  conda create -n $ENV_NAME pip python==3.8 -y
fi

set -e
conda activate $ENV_NAME
conda install $PYTORCH torchvision -c pytorch -y
pip install pytest
pip install --upgrade torchac --no-cache-dir
python -c "import torchac"
python -m pytest $TESTS_PATH -s


