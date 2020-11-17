#!/bin/bash

for VERSION in 1.4 1.5 1.6 1.7; do
  echo "Testing pytorch=$VERSION..."
  conda env remove -n torchac_test
  bash pypi/test.sh tests/ $VERSION
  python examples/mnist_autoencoder/mnist_autoencoder_example.py \
    --max_training_itr 5
done
