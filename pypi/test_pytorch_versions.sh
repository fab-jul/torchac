#!/bin/bash

set -e

source ~/Documents/miniconda3/etc/profile.d/conda.sh

LOGFILE=tested.txt
echo "Test Log" > $LOGFILE

for VERSION in 1.5 1.6 1.7; do
  echo "Testing pytorch=$VERSION..."
  rm -rf ~/.cache/torch_extensions
  rm -rf data/MNIST  # MNIST may lead to conflicts
  bash pypi/test.sh tests/test.py $VERSION
  echo "Tests pass for $VERSION" >> $LOGFILE
  conda activate torchac_test
  pip install matplotlib
  CUDA_VISIBLE_DEVICES="" python -u \
    examples/mnist_autoencoder/mnist_autoencoder_example.py \
    --max_training_itr 5
  conda deactivate
  conda env remove -n torchac_test -y
  echo "Example runs for $VERSION" >> $LOGFILE
done

echo ""
cat $LOGFILE
rm $LOGFILE
