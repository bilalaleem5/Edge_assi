# Assignment 03 – Edge Computing & Intelligent Systems
**Bilal Aleem – i221866 – Section DS-A**

PyTorch model-optimization pipeline on CIFAR-10 (ResNet-18): FP32 baseline, PTQ, QAT, unstructured pruning,
structured channel pruning, unified inference, automated benchmarking, ONNX export and ONNX Runtime CPU inference.
Full details, results and screenshots: `Technical_Report.docx`.

## Results (CIFAR-10 test set, 10,000 images, single CPU thread)
| Model | Acc. (%) | Size (MB) | MACs (M) | Latency bs=1 (ms) |
|---|---|---|---|---|
| FP32 baseline | 83.54 | 2.86 | 35.05 | 1.43 |
| PTQ INT8 | 83.53 | 0.77 | 35.05 | 0.44 |
| QAT INT8 | 83.93 | 0.77 | 35.05 | 0.43 |
| Unstructured pruned (80% sparse) | 83.96 | 2.86 (0.77 gzip) | 35.05 | 1.64 |
| Structured pruned (50% inner channels) | 82.99 | 1.48 | 17.94 | 1.25 |

Scope notes: the ResNet-18 keeps the standard topology but uses base width 16 (not 64) and 10 training epochs, because it was
trained on one CPU core; the dataset came from a GitHub image mirror of CIFAR-10 (JPEG files). See the report, Section 1.2.

## Folder layout
```
src/        all source code (one script per stage)
models/     trained checkpoints (FP32, PTQ-INT8, QAT-INT8, unstructured, structured) and models/onnx/*.onnx
results/    benchmark_results.csv, per_sample_predictions.csv, onnx_benchmark_results.csv
logs/       console output of every stage
docs/       figures used in the report
run_all.sh  full pipeline
```

## Requirements
Python 3.10+, `torch`, `torchvision`, `onnx`, `onnxruntime`, `numpy`, `pandas`, `pillow`, `matplotlib` (figures only).
```
pip install torch torchvision onnx onnxruntime numpy pandas pillow matplotlib
```
Quantized models use the `fbgemm` backend (x86 CPU).

## Run with the included checkpoints (no training)
```
python src/inference.py --model all --n 5               # predictions of all five PyTorch models
python src/inference.py --model qat --image my_img.png  # classify an image
python src/benchmark.py                                  # regenerates results/benchmark_results.csv + per_sample_predictions.csv
python src/onnx_inference.py                             # ONNX Runtime CPU benchmark -> results/onnx_benchmark_results.csv
```
(These need `data/cifar10_test.npz`; create it with step 0 below.)

## Reproduce everything from scratch
0. Data: `python src/prepare_data_torchvision.py` (downloads official CIFAR-10 into `data/`).
1. `bash run_all.sh`, or run the stages one by one:
```
python src/train_baseline.py        # FP32 baseline        -> models/resnet18_fp32.pth
python src/ptq.py                   # PTQ                  -> models/resnet18_ptq_int8.pt
python src/qat.py                   # QAT                  -> models/resnet18_qat_int8.pt
python src/prune_unstructured.py    # 80% iterative pruning-> models/resnet18_unstructured_pruned.pth
python src/prune_structured.py      # channel pruning      -> models/resnet18_structured_pruned.pth
python src/export_onnx.py           # ONNX models          -> models/onnx/
python src/benchmark.py             # PyTorch benchmark    -> results/
python src/onnx_inference.py        # ONNX benchmark       -> results/
```
Training scripts checkpoint to `state/` and resume if interrupted; set `BUDGET=<seconds>` to make a run stop and save after that time.
Seed 42 is used throughout; re-training on a different machine/dataset copy will give slightly different numbers.
