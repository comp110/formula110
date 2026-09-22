#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl
rm -rf /var/lib/apt/lists/*

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
export UV_PYTHON_INSTALL_DIR=/opt/formula110-python
uv python install 3.11
chmod -R a+rX /opt/formula110-python
uv venv --python 3.11 --managed-python /autograder/venv
uv pip install --python /autograder/venv/bin/python \
  panda3d==1.10.13 \
  ursina==7.0.0

if ! id student >/dev/null 2>&1; then
  adduser student --no-create-home --disabled-password --gecos ""
fi

install -d -m 0555 /opt/formula110-runtime
cp -R /autograder/source/trusted/racing /opt/formula110-runtime/racing
find /opt/formula110-runtime -type d -exec chmod 0555 {} +
find /opt/formula110-runtime -type f -exec chmod 0444 {} +
printf '%s\n' /opt/formula110-runtime > /autograder/venv/lib/python3.11/site-packages/formula110-runtime.pth

install -d -m 0511 /opt/formula110-exercise-autograder
install -m 0500 /autograder/source/grade.py /opt/formula110-exercise-autograder/grade.py
install -m 0500 /autograder/source/progression_worker.py /opt/formula110-exercise-autograder/progression_worker.py
install -m 0400 /autograder/source/race_worker.py /opt/formula110-exercise-autograder/race_worker.py
install -m 0555 /autograder/source/control_worker.py /opt/formula110-exercise-autograder/control_worker.py
install -m 0400 /autograder/source/config.json /opt/formula110-exercise-autograder/config.json

# The grading driver has all configuration it needs under /opt. Student code
# should not be able to inspect or modify the original autograder source.
chmod 0700 /autograder/source
