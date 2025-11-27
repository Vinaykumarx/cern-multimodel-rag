#!/usr/bin/env bash
set -e

echo "[Clean] Removing LanceDB directory ./lancedb"
rm -rf lancedb

echo "[Clean] Removing __pycache__"
find . -name "__pycache__" -type d -print -exec rm -rf {} +

echo "[Clean] Clearing Streamlit cache (~/.streamlit/cache)"
rm -rf ~/.streamlit/cache

echo "[Clean] Done."
