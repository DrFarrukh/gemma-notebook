#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
podman compose up --build -d
echo "Gemma Notebook is starting at http://127.0.0.1:8787"
echo "Run ./logs.sh to follow startup and model activity."

