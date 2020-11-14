from distutils.version import LooseVersion, StrictVersion
import argparse
import json


def main():
  p = argparse.ArgumentParser()
  p.add_argument('new_version')
  p.add_argument('--set-used', action='store_true')
  flags = p.parse_args()
  new_version = flags.new_version
  set_used = flags.set_used

  with open('bin/version.json', 'r') as f:
    version_info = json.load(f)

  is_unused = not version_info["used"]
  cur_version = version_info["version"]
  if is_unused:
    version_is_ok = LooseVersion(new_version) >= LooseVersion(cur_version)
  else:
    version_is_ok = LooseVersion(new_version) > LooseVersion(cur_version)

  if not version_is_ok:
    raise ValueError(f'Not ok: {new_version}. Current version={cur_version}')

  with open('bin/version.json', 'w') as f:
    json.dump({'version': new_version, 'used': set_used}, f)


if __name__ == '__main__':
  main()



