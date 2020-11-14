#!/bin/bash
#
# Test PyPi package
#

TESTS_PATH=$1

if [[ -z $TESTS_PATH ]]; then
    echo "Usage: $0 TESTS_PATH"
    exit 1
fi

source /Users/fabian/Documents/miniconda3/etc/profile.d/conda.sh
conda create -n torchac_test pip python==3.8 -y
conda activate torchac_test
conda install pytorch torchvision -c pytorch -y
pip install pytest
pip install --upgrade torchac --no-cache-dir
python -c "import torchac"
python -m pytest $TESTS_PATH -s


