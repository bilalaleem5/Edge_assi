"""Generate result charts for the report from the benchmark CSVs."""
import os, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import RESULT_DIR
d = pd.read_csv(os.path.join(RESULT_DIR, 'benchmark_results.csv'))
short = ['FP32', 'PTQ INT8', 'QAT INT8', 'Unstruct.\n80%', 'Struct.\nchannels']
col = ['#4C72B0', '#55A868', '#2E8B57', '#C44E52', '#DD8452']
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a, (c, t) in zip(ax, [('test_accuracy_%', 'Test accuracy (%)'), ('file_size_MB', 'Checkpoint size (MB)'),
                          ('latency_bs1_mean_ms', 'Latency, batch 1 (ms)'), ('MACs_M', 'MACs (millions)')]):
    b = a.bar(short, d[c], color=col); a.set_title(t, fontsize=10); a.tick_params(labelsize=8)
    if c == 'test_accuracy_%': a.set_ylim(75, 86)
    for r, v in zip(b, d[c]): a.text(r.get_x() + r.get_width() / 2, v, f'{v:g}', ha='center', va='bottom', fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(RESULT_DIR), 'docs', 'fig_benchmark.png'), dpi=160)
o = pd.read_csv(os.path.join(RESULT_DIR, 'onnx_benchmark_results.csv'))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); n = ['FP32', 'PTQ', 'QAT', 'Unstr.', 'Struct.']
ax[0].bar(n, o['latency_bs1_mean_ms'], color=col); ax[0].set_title('ONNX Runtime CPU latency, batch 1 (ms)', fontsize=10)
ax[1].bar(n, o['test_accuracy_%'], color=col); ax[1].set_ylim(75, 86); ax[1].set_title('ONNX test accuracy (%)', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(RESULT_DIR), 'docs', 'fig_onnx.png'), dpi=160)
