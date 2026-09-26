#!/usr/bin/env bash
# Install Docker on your instance and let your user run it without sudo.
# Safe to run twice.
set -euo pipefail

if docker info >/dev/null 2>&1; then
  echo "Docker is already installed and running."
  docker --version
  exit 0
fi

echo "Installing Docker..."
if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y docker
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y docker
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y docker.io
else
  echo "No dnf, yum or apt-get found. Install Docker by hand, then re-run ./serve.sh" >&2
  exit 1
fi

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo
echo "Docker installed."
if docker info >/dev/null 2>&1; then
  echo "Ready. Next: ./serve.sh"
else
  echo "You were added to the docker group, which only takes effect on a new login."
  echo "Run this, then ./serve.sh :"
  echo "    newgrp docker"
fi
