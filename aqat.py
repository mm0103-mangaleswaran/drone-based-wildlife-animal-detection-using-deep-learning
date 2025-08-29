
import torch
import torch.nn as nn
from typing import Dict, Any

def fake_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Simple symmetric fake quantization to [-1, 1] with STE."""
    if bits is None or bits >= 16:
        return x
    scale = (2 ** bits - 1)
    # Map to [-1,1] with tanh to stabilize
    x_t = torch.tanh(x)
    q = torch.round((x_t + 1.0) * 0.5 * scale) / scale * 2.0 - 1.0
    return x + (q - x).detach()

class AqatManager:
    """Registers hooks to collect activation variances and apply per-module fake-quantization."""
    def __init__(self, model: nn.Module, var_threshold: float = 0.5):
        self.model = model
        self.var_threshold = var_threshold
        self.handles = []
        self.stats: Dict[nn.Module, Dict[str, Any]] = {}
        self.bits: Dict[nn.Module, int] = {}

    def _forward_hook(self, module, inp, out):
        with torch.no_grad():
            var = out.var().item() if isinstance(out, torch.Tensor) else 1.0
            self.stats.setdefault(module, {}).setdefault("vars", []).append(var)
        # Apply activation quantization on forward output based on assigned bits
        b = self.bits.get(module, None)
        if b is not None and isinstance(out, torch.Tensor) and self.model.training:
            return fake_quantize(out, b)
        return None  # use original output

    def _pre_forward_hook(self, module, inp):
        # Quantize weights on the fly
        b = self.bits.get(module, None)
        if b is not None and hasattr(module, 'weight') and self.model.training:
            module._cached_weight = module.weight.data.clone()
            module.weight.data = fake_quantize(module.weight.data, b)

    def _pre_forward_hook_restore(self, module, inp, result):
        # Restore original weights after forward
        if hasattr(module, '_cached_weight'):
            module.weight.data = module._cached_weight
            delattr(module, '_cached_weight')

    def attach(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.handles.append(m.register_forward_pre_hook(self._pre_forward_hook))
                self.handles.append(m.register_forward_hook(self._forward_hook))
                self.handles.append(m.register_forward_hook(self._pre_forward_hook_restore))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def assign_bits(self):
        """Assign 4 or 8 bits per module based on running activation variance."""
        for m, st in self.stats.items():
            if 'vars' not in st or len(st['vars']) == 0:
                continue
            v = sum(st['vars']) / len(st['vars'])
            self.bits[m] = 4 if v < self.var_threshold else 8

def enable_aqat(model: nn.Module, var_threshold: float = 0.5) -> AqatManager:
    mgr = AqatManager(model, var_threshold=var_threshold)
    mgr.attach()
    return mgr

def disable_aqat(mgr: AqatManager):
    mgr.detach()
