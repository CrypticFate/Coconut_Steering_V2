#!/usr/bin/env bash
set -euo pipefail

mkdir -p external
if [ ! -d "external/coconut" ]; then
  git clone https://github.com/facebookresearch/coconut.git external/coconut
fi

mkdir -p data/coconut_format
cd external/coconut
bash preprocessing/gsm_icot.bash
mv data/gsm_train.json ../../data/coconut_format/
mv data/gsm_valid.json ../../data/coconut_format/
mv data/gsm_test.json ../../data/coconut_format/
cd ../..

echo "Setup complete:"
ls -lh data/coconut_format/
