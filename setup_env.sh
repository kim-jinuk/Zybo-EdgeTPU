#!/usr/bin/env bash
# Ubuntu / PetaLinux ARM
set -e
sudo apt update && sudo apt install -y python3-venv git cmake g++ pkg-config \
    libopencv-dev libedgetpu1-std python3-scipy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt