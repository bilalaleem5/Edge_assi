#!/usr/bin/env bash
# Reproducible end-to-end pipeline (CPU only). Run from the project root:  bash run_all.sh
set -e
PY=${PYTHON:-python}
[ -f data/cifar10_train.npz ] || { echo "Missing data: run  $PY src/prepare_data_torchvision.py"; exit 1; }
mkdir -p models results logs
$PY src/train_baseline.py          2>&1 | tee logs/train_baseline.log
$PY src/ptq.py                     2>&1 | tee logs/ptq.log
$PY src/qat.py                     2>&1 | tee logs/qat.log
$PY src/prune_unstructured.py      2>&1 | tee logs/prune_unstructured.log
$PY src/prune_structured.py        2>&1 | tee logs/prune_structured.log
$PY src/export_onnx.py             2>&1 | tee logs/export_onnx.log
$PY src/benchmark.py               2>&1 | tee logs/benchmark.log
$PY src/onnx_inference.py          2>&1 | tee logs/onnx_benchmark.log
$PY src/inference.py --model all --n 5
