#!/bin/bash

# 데이터 파이프라인 검증 스크립트
# Python 스크립트로 모든 인자를 전달

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Run validation
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Pass all arguments to Python script
python training/src/validate/pipeline.py "$@"
