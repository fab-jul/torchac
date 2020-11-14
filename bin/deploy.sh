#!/bin/bash

set -e

VERSION_NUMBER=$1

if [[ -z $VERSION_NUMBER ]]; then
  echo "Usage: $0 VERSION_NUMBER"
  exit 1
fi

if [ -z "$(git status --porcelain)" ]; then
  echo "Git clean"
else
  echo "Git not clean, please commit"
  exit 1
fi

exit 0

python bin/update_version.py $VERSION_NUMBER

rm -rf dist/
python setup.py bdist_wheel
twine upload dist/*

python bin/update_version.py $VERSION_NUMBER --set-used

bash pypi/test.sh tests/test.py
