"""Render terminal-output screenshots (PNG) of the key log files for the technical report."""
import os, re, textwrap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); L = os.path.join(R, 'logs'); O = os.path.join(R, 'docs', 'shots'); os.makedirs(O, exist_ok=True)
JUNK = re.compile(r'warn|migrat|torchao|eager mode|FX graph|pt2e|issues/2259|qweight|convert\(|prepare|super\(\)|\[resume|\[budget|^\s*$', re.I)
def shot(log, name, title, last=None, first=None, maxlines=16):
    ls = [l.rstrip() for l in open(os.path.join(L, log)) if not JUNK.search(l)]
    seen = []; [seen.append(l) for l in ls if l not in seen]
    ls = seen
    if first: ls = ls[:first]
    if last: ls = ls[-last:]
    ls = [t for l in ls[:maxlines] for t in (textwrap.wrap(l, 118, subsequent_indent='    ') or [''])]
    fig = plt.figure(figsize=(10, 0.28 * (len(ls) + 2) + 0.2)); fig.patch.set_facecolor('#1e1e1e')
    fig.text(0.01, 1 - 0.28 / (0.28 * (len(ls) + 2) + 0.2), f'$ {title}', color='#6ad46a', family='monospace', fontsize=8.5, va='center')
    for i, l in enumerate(ls):
        fig.text(0.01, 1 - 0.28 * (i + 2) / (0.28 * (len(ls) + 2) + 0.2), l, color='#e6e6e6', family='monospace', fontsize=8.5, va='center')
    fig.savefig(os.path.join(O, name), dpi=140, facecolor=fig.get_facecolor()); plt.close(fig)
shot('train_baseline.log', 'baseline.png', 'python src/train_baseline.py', last=12, maxlines=12)
shot('ptq.log', 'ptq.png', 'python src/ptq.py')
shot('qat.log', 'qat.png', 'python src/qat.py')
shot('prune_unstructured.log', 'unstructured.png', 'python src/prune_unstructured.py', maxlines=24)
shot('prune_structured.log', 'structured.png', 'python src/prune_structured.py', last=10)
shot('benchmark.log', 'benchmark.png', 'python src/benchmark.py', last=14)
shot('onnx_benchmark.log', 'onnx.png', 'python src/onnx_inference.py', last=7)
shot('inference_demo.log', 'inference.png', 'python src/inference.py --model all --n 3', maxlines=18)
