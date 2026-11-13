"""Export every model to ONNX (models/onnx/).
 FP32 / unstructured / structured : torch.onnx.export of the float network
 PTQ INT8                         : ONNX Runtime static quantization (QDQ) of the FP32 ONNX graph, calibrated on train data
 QAT INT8                         : fake-quantized QAT network exported as QuantizeLinear/DequantizeLinear (QDQ) nodes"""
import os, warnings, torch, numpy as np, onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from common import *
from inference import load_model

warnings.filterwarnings('ignore')
OUT = os.path.join(MODEL_DIR, 'onnx'); os.makedirs(OUT, exist_ok=True)
dummy = torch.randn(1, 3, 32, 32)


def export_float(model, path):
    model.eval()
    torch.onnx.export(model, dummy, path, input_names=['image'], output_names=['logits'],
                      dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=13, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    print(f'exported {os.path.basename(path)} ({file_mb(path):.2f} MB)')


class Reader(CalibrationDataReader):
    def __init__(self, n=500):
        x, y = load_split('train'); g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(x.size(0), generator=g)[:n]
        self.data = iter([{'image': normalize(x[i:i + 1]).numpy()} for i in idx.tolist()])
    def get_next(self):
        return next(self.data, None)


export_float(load_model('fp32'), f'{OUT}/resnet18_fp32.onnx')
export_float(load_model('unstructured'), f'{OUT}/resnet18_unstructured_pruned.onnx')
export_float(load_model('structured'), f'{OUT}/resnet18_structured_pruned.onnx')

quantize_static(f'{OUT}/resnet18_fp32.onnx', f'{OUT}/resnet18_ptq_int8.onnx', Reader(), quant_format=QuantFormat.QDQ,
                per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
print(f'exported resnet18_ptq_int8.onnx ({file_mb(f"{OUT}/resnet18_ptq_int8.onnx"):.2f} MB)')

# QAT: rebuild the fake-quant graph, load the fine-tuned weights/observer ranges and export QDQ nodes
m = ResNet18(); m.train(); m.fuse_model()
m.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm'); torch.ao.quantization.prepare_qat(m, inplace=True)
m.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'resnet18_qat_int8_fakequant_state_dict.pth'))); m.eval()


def _plain_fq(self, x):
    """Observers are frozen after QAT, so use the plain (ONNX-exportable) fake-quant op with the learned scale / zero-point."""
    if self.is_per_channel:
        return torch.fake_quantize_per_channel_affine(x, self.scale, self.zero_point, self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
    return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, self.activation_post_process.quant_min, self.activation_post_process.quant_max)


for mod in m.modules():
    if isinstance(mod, torch.ao.quantization.FakeQuantizeBase) and hasattr(mod, 'is_per_channel'):
        mod.forward = _plain_fq.__get__(mod)
export_float(m, f'{OUT}/resnet18_qat_int8.onnx')
