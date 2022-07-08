#!/bin/bash

TEST_DIR=_torchac_test_
if [[ ! -f tests/test.py ]]; then
	echo "Expected tests/test.py, are your running from repo root?"
	exit 1
fi
if [[ -d $TEST_DIR ]]; then 
	echo "Exists! $TEST_DIR"
	exit 1
fi
mkdir -p $TEST_DIR
pushd $TEST_DIR
python3 -m venv .venv
. ./.venv/bin/activate
pip install torch numpy ninja torch pytest
popd
pip list
python -m pytest tests/test.py
deactivate
rm -rf $TEST_DIR

