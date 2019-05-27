#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..

mkdir -p ./output

python -u  core/train.py train_raw > check_27.log 2>&1