#!/usr/bin/env bash
#
# Launches AudioLDM 2 Web UI.

set -e

deactivate > /dev/null 2>&1 || true

if [ ! -f ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install \
  --pre \
  --extra-index-url "https://download.pytorch.org/whl/nightly/cpu" \
  -r requirements.txt

export PYTORCH_ENABLE_MPS_FALLBACK=1

streamlit run webui.py
