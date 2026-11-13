"""CPU inference + benchmark for the exported ONNX models (ONNX Runtime, CPUExecutionProvider).
Usage:  python src/onnx_inference.py                      -> benchmark all ONNX models, writes results/onnx_benchmark_results.csv
        python src/onnx_inference.py --image img.png --model ptq_int8"""
import argparse, os, time, numpy as np, pandas as pd, onnxruntime as ort
from PIL import Image
from common import *

ONNX = {'fp32': 'resnet18_fp32.onnx', 'ptq_int8': 'resnet18_ptq_int8.onnx', 'qat_int8': 'resnet18_qat_int8.onnx',
        'unstructured_pruned': 'resnet18_unstructured_pruned.onnx', 'structured_pruned': 'resnet18_structured_pruned.onnx'}


def session(name):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    if name == 'qat_int8':      # QDQ graph exported from QAT: keep Q/DQ nodes explicit (avoid QLinear fusion of per-tensor-ambiguous residual adds)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(os.path.join(MODEL_DIR, 'onnx', ONNX[name]), so, providers=['CPUExecutionProvider'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image'); ap.add_argument('--model', default='fp32', choices=list(ONNX))
    a = ap.parse_args()
    if a.image:
        im = Image.open(a.image).convert('RGB').resize((32, 32))
        x = normalize(__import__('torch').from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0)).numpy()
        out = session(a.model).run(None, {'image': x})[0][0]
        p = np.exp(out - out.max()); p /= p.sum()
        print(f'{a.model}: {CLASSES[int(p.argmax())]} (confidence {p.max():.2f})')
        raise SystemExit
    xt, yt = load_split('test'); X = normalize(xt).numpy(); Y = yt.numpy(); rows = []
    for name in ONNX:
        s = session(name); t0 = time.time(); pr = []
        for i in range(0, len(X), 250):
            pr.append(s.run(None, {'image': X[i:i + 250]})[0].argmax(1))
        tt = time.time() - t0; acc = 100 * (np.concatenate(pr) == Y).mean()
        lat = []
        for k in range(350):
            t = time.perf_counter(); s.run(None, {'image': X[k:k + 1]}); d = (time.perf_counter() - t) * 1000
            if k >= 50: lat.append(d)
        path = os.path.join(MODEL_DIR, 'onnx', ONNX[name])
        rows.append({'onnx_model': ONNX[name], 'test_accuracy_%': round(acc, 2), 'file_size_MB': round(file_mb(path), 3),
                     'latency_bs1_mean_ms': round(float(np.mean(lat)), 3), 'latency_bs1_p95_ms': round(float(np.percentile(lat, 95)), 3),
                     'throughput_img_per_s': round(len(X) / tt, 1)})
        print(rows[-1])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULT_DIR, 'onnx_benchmark_results.csv'), index=False)
    print(df.to_string(index=False))
