import setuptools
import json


def _get_long_description():
  with open('README.md', 'r') as f:
    long_description_lines = []
    skip = False
    for line in f:
      if '<div' in line:
        skip = True
      if '</div>' in line:
        skip = False
      if skip:
        continue
      long_description_lines.append(line)
    return ''.join(long_description_lines)


def _get_version():
  with open('bin/version.json', 'r') as f:
    version_info = json.load(f)
  if version_info['used']:
    raise ValueError('Version already used!')
  return version_info['version']


setuptools.setup(
  name='torchac',
  packages=['torchac'],
  version=_get_version(),
  author='fab-jul',
  author_email='fabianjul@gmail.com',
  description='Fast Arithmetic Coding for PyTorch',
  long_description=_get_long_description(),
  long_description_content_type='text/markdown',
  python_requires='>=3.6',
  license='GNU General Public License',
  url='https://github.com/fab-jul/torchac')
