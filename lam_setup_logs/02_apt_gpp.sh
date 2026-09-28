#!/usr/bin/env bash
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get install -y g++ 2>&1
echo "EXIT_GPP=$?"
g++ --version | head -1
which g++ c++
