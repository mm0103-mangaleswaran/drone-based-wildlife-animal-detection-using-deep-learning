
# MobileYOLO-Lite (Module-wise Code)

This repository provides a modular PyTorch implementation of **MobileYOLO-Lite: Efficient Real-Time Tiny Animal Detection in Drone Imagery**. It follows the architecture described in the attached article with five core components:

1. **PatchPack** (2×2 spatial-to-channel reshape)
2. **Ghost-Mobile Backbone** (GhostConv + linear bottlenecks)
3. **Lite-BiDFPN** (lightweight bidirectional weighted fusion)
4. **Tiny-Head with DynamicSA** (dual heads at 152×152 and 76×76 with sparse attention gates)
5. **AQAT-aware quantization & SGDM-AQS optimizer**

## Structure
```text
mobileyolo_lite/
  mobileyolo_lite/
    __init__.py
    config.py
    patchpack.py
    ghost.py
    backbone.py
    lite_bidfpn.py
    dynamic_sa_head.py
    aqat.py
    sgdm_aqs.py
    model.py
    nms.py
    utils.py
  examples/
    train_stub.py
  README.md
```

## Quick Start
```bash
pip install torch  # (PyTorch >= 1.12)
cd /path/to/mobileyolo_lite
python -m examples.train_stub
```

Replace the **dummy loss** in `examples/train_stub.py` with your YOLO loss and wire up a dataloader for WAID.

## Notes
- The sparse attention gate masks activations to approximate conditional compute in PyTorch.
- AQAT uses simple STE fake-quantization with per-module **4-bit or 8-bit** assignment based on activation variance.
- `SGDMAQS` implements dual momentum and gradient filtering via a `gate_factor` scalar emitted by the head.
- Anchors and thresholds are defined in `config.py` and can be tuned for your dataset.
