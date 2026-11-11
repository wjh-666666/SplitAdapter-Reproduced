#!/usr/bin/env bash
# Fetch the third-party dependencies into third_party/ (they are not redistributed with this repository).
#   bash scripts/setup_third_party.sh
# Isaac Gym Preview 4 has to be downloaded manually from NVIDIA (license agreement):
#   https://developer.nvidia.com/isaac-gym  ->  extract to third_party/isaacgym
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
PHYSHSI_COMMIT=07bf22bdaf9c23c1d40366334a744c1816b6d42b   # version used for all results in this repository
if [ ! -d third_party/PhysHSI ]; then
  git clone https://github.com/InternRobotics/PhysHSI.git third_party/PhysHSI
fi
git -C third_party/PhysHSI checkout -q $PHYSHSI_COMMIT
if [ ! -d third_party/isaacgym/python ]; then
  echo "Please download Isaac Gym Preview 4 and extract it to third_party/isaacgym (see README)."
  exit 1
fi
pip install -e third_party/isaacgym/python
pip install -e third_party/PhysHSI/rsl_rl -e third_party/PhysHSI/legged_gym
pip install -r requirements.txt
echo "done"
