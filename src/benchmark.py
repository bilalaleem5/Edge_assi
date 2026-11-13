"""Automated benchmark of all five PyTorch models on the full CIFAR-10 test set.
Outputs:  results/benchmark_results.csv        (one row per model: accuracy, size, params, MACs, latency, throughput)
          results/per_sample_predictions.csv   (per-image predictions + confidence of every model)"""
import os, time, io, gzip, torch, numpy as np, pandas as pd
from common import *
from inference import MODELS, load_model, model_path

torch.set_num_threads(1)      # single thread -> fair, reproducible CPU comparison
xt, yt = load_split('test'); X = normalize(xt)
rows, pred_cols = [], {}
pred_cols['index'] = np.arange(len(yt)); pred_cols['true_label'] = [CLASSES[i] for i in yt.tolist()]

for name, (fname, label) in MODELS.items():
    m = load_model(name)
    preds, confs = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(X), 250):
            p = torch.softmax(m(X[s:s + 250]), 1); c, i = p.max(1); preds.append(i); confs.append(c)
    total_t = time.time() - t0
    preds = torch.cat(preds); confs = torch.cat(confs)
    acc = 100.0 * (preds == yt).float().mean().item()
    # batch-1 latency (50 warm-up + 300 timed runs)
    lat = []
    with torch.no_grad():
        for k in range(350):
            t = time.perf_counter(); m(X[k:k + 1]); dt = (time.perf_counter() - t) * 1000
            if k >= 50: lat.append(dt)
    is_q = name in ('ptq', 'qat')
    if is_q:
        params = nz = None; macs = None
        # parameters of quantized model = int8 weights; count from the FP32 twin for reference
        fp = load_model('fp32'); params = count_params(fp); nz = params; macs = count_macs(fp)
    else:
        params = count_params(m); nz = count_nonzero(m); macs = count_macs(m)
    p = model_path(name)
    gz = None
    if not is_q:
        buf = io.BytesIO(); torch.save(m.state_dict(), buf); gz = len(gzip.compress(buf.getvalue())) / 1e6
    rows.append({
        'model': label, 'checkpoint': fname, 'test_accuracy_%': round(acc, 2),
        'file_size_MB': round(file_mb(p), 3), 'gzip_size_MB': round(gz, 3) if gz else round(file_mb(p), 3),
        'parameters': params, 'nonzero_parameters': nz, 'sparsity_%': round(100 * (1 - nz / params), 2),
        'MACs_M': round(macs / 1e6, 2), 'latency_bs1_mean_ms': round(float(np.mean(lat)), 3),
        'latency_bs1_p95_ms': round(float(np.percentile(lat, 95)), 3),
        'throughput_img_per_s': round(len(X) / total_t, 1)})
    pred_cols[f'{name}_pred'] = [CLASSES[i] for i in preds.tolist()]
    pred_cols[f'{name}_conf'] = np.round(confs.numpy(), 4)
    print(f'{label:<30} acc {acc:.2f}%  lat {np.mean(lat):.2f} ms  size {file_mb(p):.2f} MB')

df = pd.DataFrame(rows); base = df.iloc[0]
df['speedup_vs_fp32'] = (base['latency_bs1_mean_ms'] / df['latency_bs1_mean_ms']).round(2)
df['size_reduction_vs_fp32_x'] = (base['file_size_MB'] / df['file_size_MB']).round(2)
df['accuracy_delta_vs_fp32'] = (df['test_accuracy_%'] - base['test_accuracy_%']).round(2)
os.makedirs(RESULT_DIR, exist_ok=True)
df.to_csv(os.path.join(RESULT_DIR, 'benchmark_results.csv'), index=False)
pd.DataFrame(pred_cols).to_csv(os.path.join(RESULT_DIR, 'per_sample_predictions.csv'), index=False)
print('\n', df[['model', 'test_accuracy_%', 'file_size_MB', 'sparsity_%', 'MACs_M', 'latency_bs1_mean_ms', 'speedup_vs_fp32']].to_string(index=False))
