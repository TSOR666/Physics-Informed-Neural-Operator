# Enhanced PINO (Production‑Ready Physics‑Informed Neural Operator)

A robust **3D** physics‑informed neural operator (PINO/FNO) for volumetric fields with **adaptive spectral convolution**, **physics validators**, and **production niceties** (mixed precision, `torch.compile`, health/perf monitoring, LRU caches, safe fallbacks). Spatial‑only (3D) in this release; time‑dependent constraints can be added via state sequences.

---

## ✨ Highlights
- **3D Fourier Neural Operator core** with **adaptive mode selection** and FP32 FFTs for stability; automatic **Conv3D fallback** if spectral path degrades.
- **Physics constraints** you can turn on per‑task:
  - Incompressible flow (Navier–Stokes: \(\nabla\cdot u=0\)).
  - Helmholtz \((\nabla^2 + k^2)u = 0)\) and Poisson \((\nabla^2 u = f)\).
  - Radiative transfer proxy via optical depth \(\tau=-\log I\) with \(|\nabla\tau|\approx \kappa\).
  - Optional **Smagorinsky** turbulence closure penalty.
  - **Conservation** penalties: energy, momentum, mass (compare input vs. output scalars).
- **Spherical Harmonics encoder (optional)** for directional conditioning.
- **Mixed precision** (BF16 recommended on Ampere+), **TF32** enabled on NVIDIA by default, `torch.compile` hook, and optional **`channels_last_3d`** memory layout.
- **Health monitor** (spectral degradation counts/energy) and **performance tracker** (latency/memory) with bounded histories.
- **Spectral regularization**: \(k^2\)-weighted penalty to control high‑frequency noise.

---

## 📦 Installation
Requires **PyTorch ≥ 2.0**, **NumPy**. GPU strongly recommended. BF16 autocast supported on Ampere+.

```bash
pip install torch numpy
```

> TIP: On Ampere+ (A100/RTX 30xx/40xx/H100), BF16 is recommended. TF32 matmul/conv is enabled by default.

---

## 🏁 Quickstart

```python
import torch
from pino import EnhancedPINO, EnhancedPINOConfig, PhysicsConstraintType as P

# Configure
cfg = EnhancedPINOConfig(
    spatial_dims=(64, 64, 32),     # [D, H, W]
    input_channels=4,              # e.g., velocity(u,v,w) + pressure
    num_spectral_bands=64,         # output feature channels
    adaptive_modes=True,

    # Physics you want
    physics_constraints=[
        P.ENERGY_CONSERVATION,
        P.MOMENTUM_CONSERVATION,
        P.NAVIER_STOKES,
        P.RADIATIVE_TRANSFER,
    ],
    constraint_weights={
        "energy_conservation": 1.0,
        "momentum_conservation": 0.5,
        "navier_stokes": 1.0,
        "radiative_transfer": 0.25,
    },
    velocity_channels=[0,1,2],    # u,v,w live in output channels 0..2

    # Performance
    mixed_precision=True,          # BF16 auto‑enabled on Ampere+
    amp_dtype=torch.bfloat16,
    compile_model=True,
)

model = EnhancedPINO(cfg)
model.enable_channels_last_3d()      # optional
model.eval()

# Inputs
B = 2
x = torch.randn(B, 4, *cfg.spatial_dims, device=cfg.device) * 0.1
inputs = {
    "state": x,
    "directions": torch.nn.functional.normalize(torch.randn(B,3, device=cfg.device), dim=-1),
    # Optional conservation targets (scalars/vectors per sample)
    "energy": torch.ones(B,1, device=cfg.device),
    "momentum": torch.zeros(B,3, device=cfg.device),
    "mass": torch.ones(B,1, device=cfg.device)*10.0,
}

with torch.no_grad():
    outputs = model(inputs)

# Physics loss (for training)
loss_phys = model.compute_physics_loss(outputs, inputs)
```

**Tensor shapes**
- `state`: `[B, C, D, H, W]` (float) — input field(s)
- `directions` (optional): `[B, 3]` — unit vectors for SH encoder
- Outputs include `state` (predicted field), `energy/momentum/mass`, optional `health_report`, and perf stats.

---

## 🧠 Architecture

### EnhancedFNO3D backbone
- **Input projection**: `Conv3d(C_in → W)` + GELU.
- **Fourier blocks** (× `num_layers`):
  1. **AdaptiveSpectralConvolution3D** on FP32 FFTs (rFFT on last axis), with **learned mode selector** and safe **active‑mode capping**.
  2. Local `1×1×1 Conv3d` branch.
  3. Residual sum + GELU.
- **Output projection**: `Conv3d(W → W/2) → GELU → Conv3d(W/2 → num_spectral_bands)`.
- **Optional SH encoder** projects real spherical‑harmonic features of unit direction vectors and fuses them into the operator features.

### Adaptive spectral convolution (ASC3D)
Let \(X\in\mathbb{R}^{B\times C_{in}\times D\times H\times W}\). ASC3D computes

1. **Forward FFT** (cast to FP32): \(\mathcal{F}\{X\} = X_\omega\).
2. **Mode selection**: choose active bands `(d,h,w)` by a sigmoid‑gated vector and `mode_fraction`; respect rFFT half‑spectrum on the last axis.
3. **Complex spectral weights** \(W\in\mathbb{C}^{C_{out}\times C_{in}\times d\times h\times w}\): apply via a correct batched `einsum`.
4. **Inverse FFT** (cast back to original dtype).
5. **Fallback**: if the spectral path fails (OOM/NAN/etc.), fall back to `Conv3d` and record a degradation event.

**Monitoring**: ASC3D records spectral energy and active modes per layer; histories are bounded to avoid memory leaks.

---

## 🧪 Physics & Math (built‑in residuals)
All residuals are **mean squared** penalties over the spatial domain.

### Conservation terms
- **Energy**: \(E = \|x\|_2^2 / |\Omega|\) (normalized by spatial volume). Loss compares predicted vs. provided `inputs["energy"]`.
- **Momentum**: coordinate‑weighted sum per axis, normalized by volume; channel‑aware (uses velocity channels if set).
- **Mass**: total integral of the state (sum over spatial dims).

### PDE residuals (spatial)
- **Incompressible flow** (Navier–Stokes constraint):
  \[ \mathcal{L}_{NS} = \operatorname{mean}(\, (\nabla\cdot u)^2 \,). \]
  `velocity_channels=[0,1,2]` maps which output channels are \(u,v,w\).

- **Helmholtz**:
  \[ \mathcal{L}_{\text{Helm}} = \operatorname{mean}\,\big( (\nabla^2 u + k^2 u)^2 \big). \]

- **Poisson** (with optional source \(f\)):
  \[ \mathcal{L}_{\text{Pois}} = \operatorname{mean}\,\big( (\nabla^2 u - f)^2 \big). \]

- **Radiative transfer proxy** (optical depth): let intensity \(I=\operatorname{mean}_c x\), \(\tau=-\log I\), target gradient magnitude \(\kappa\). Penalty:
  \[ \mathcal{L}_{\text{RT}} = \operatorname{mean}\,\big(\, (\|\nabla \tau\| - \kappa)^2 \,\big). \]

- **Smagorinsky turbulence closure** (optional): compute the strain‑rate tensor and eddy viscosity \(\nu_t=(C_s\,\Delta)^2\,\sqrt{2S_{ij}S_{ij}}\); penalize pathological negative \(\nu_t\).

**Spectral Laplacian** is computed via FP32 FFTs: \(\nabla^2 u = \mathcal{F}^{-1}[-\|k\|^2\,\mathcal{F}\{u\}]\) with consistent FFT normalization.

### Spectral regularization
Penalize high‑frequency energy with a \(k^2\)-weighted term: \(\mathcal{L}_{spec} = \operatorname{mean}(\, \|k\|^2\,|\hat{x}(k)|^2 \,)\). Controlled by `spectral_regularization`.

> **Boundary conditions:** spectral Laplacians assume **periodic** BCs. For non‑periodic domains, consider padding, windowing, or swapping to a finite‑difference Laplacian.

---

## 🏋️ Training strategy
Typical loss:
\[ \mathcal{L} = \underbrace{\mathcal{L}_{\text{sup}}}_{\text{MSE/MAE on fields}} \; + \; \lambda_{phys}\,\underbrace{\sum_i w_i\,\mathcal{L}_i}_{\text{physics}} \; + \; \lambda_{spec}\,\mathcal{L}_{spec}. \]

**Recommended recipe**
1. **Start supervised** (pure data loss) for a few epochs.
2. **Turn on conservation** (energy/momentum/mass) with small weights.
3. **Add PDE terms** (NS/Helmholtz/Poisson/RT) progressively; ramp weights.
4. Enable **spectral regularization** if you see HF ringing.

**Hyper‑params to watch**
- `constraint_weights`: per‑term weights; begin with 0.1–1.0.
- `physics_loss_weight` (`λ_phys`): 0.1–1.0 typical; ramp up late.
- `velocity_channels`: required for NS/turbulence to reference the right outputs.
- `radiative_kappa`: extinction coefficient for radiative term (0.05–0.2 usually).

**Performance switches**
- `mixed_precision=True` with `amp_dtype=torch.bfloat16` on Ampere/Hopper.
- `compile_model=True` (`torch.compile`) for inference/training speedups.
- `checkpoint_gradients=True` on deep stacks to save VRAM.
- `enable_channels_last_3d()` for conv speed on newer CUDA.

**Stability**
- FFTs in spectral physics run in **FP32** (even under autocast).
- All histories are **bounded**; health monitor can alert on degradation/instability.
- `debug_strict=True` to raise if spectral path degrades repeatedly.

---

## 📊 Evaluation & monitoring
- **Physics residuals**: report each active constraint and the aggregate physics loss.
- **Conservation deltas**: \(|E_{out}-E_{in}|,\ |p_{out}-p_{in}|,\ |m_{out}-m_{in}|)\).
- **Throughput & memory**: mean/std/peak from the performance tracker.
- **Reliability**: spectral degradation counts/rates; spectral energy stability; "active modes" per layer.

---

## 🔌 Applications
- **Physics‑aware priors/regularizers** inside diffusion samplers for scientific volumes.
- **Incompressible flow** surrogates and stability‑checked rollouts.
- **Helmholtz/Poisson** PDE‑constrained reconstruction (e.g., tomography, Poisson blending in 3D).
- **Radiative transfer** proxies in rendering or remote sensing via optical‑depth control.
- **Turbulence** experiments with Smagorinsky‑style stabilization.

---

## 🧩 API sketch

### `EnhancedPINOConfig`
Key fields:
- **Architecture**: `hidden_dim`, `num_layers`, `num_spectral_bands`, `spatial_dims`, `input_channels`.
- **Spectral**: `adaptive_modes`, `mode_fraction`, `spectral_regularization`, `fft_norm`.
- **Physics**: `physics_constraints`, `constraint_weights`, `physics_loss_weight`, `velocity_channels`, `radiative_kappa`, `use_turbulence_model`, `turbulence_model`.
- **Training**: `learning_rate`, `batch_size`, `max_epochs`, `early_stopping_patience`, `gradient_clip_norm`, `gradient_accumulation_steps`.
- **Performance**: `device`, `mixed_precision`, `amp_dtype`, `compile_model`, `checkpoint_gradients`.
- **Monitoring**: `enable_health_monitoring`, `degradation_threshold`, `log_frequency`.
- **Caches & limits**: `max_cache_size`, `max_history_size`.

### `EnhancedPINO`
- `forward(inputs: Dict[str, Tensor]) -> Dict[str, Tensor]` — predicts `state` and derived scalars, plus health/perf.
- `compute_physics_loss(outputs, inputs=None) -> Tensor` — aggregates active physics and conservation penalties.
- `enable_channels_last_3d()` / `format_input(t)` — optional memory‑format helpers.

### `PhysicsConstraintType`
`ENERGY_CONSERVATION`, `MOMENTUM_CONSERVATION`, `MASS_CONSERVATION`, `NAVIER_STOKES`, `HELMHOLTZ`, `POISSON`, `RADIATIVE_TRANSFER`, `TURBULENCE_CLOSURE` (time‑dependent placeholders exist but are not implemented here).

---

## 🧰 Training loop (minimal example)

```python
opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

for step, batch in enumerate(loader):
    batch = {k: v.to(cfg.device) for k,v in batch.items()}
    with torch.autocast(cfg.device if "cuda" in str(cfg.device) else "cpu", dtype=cfg.amp_dtype, enabled=cfg.mixed_precision):
        out = model(batch)
        loss_sup = torch.nn.functional.mse_loss(out["state"], batch["target"])  # if supervised
        loss_phys = model.compute_physics_loss(out, batch)
        loss = loss_sup + loss_phys

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
```

---

## 🪪 License
See licence for more info.

---

## 🙌 Citation
If you use this PINO in academic work, please cite your project/PINO repository (add a `CITATION.cff`).

```
@software{pino_enhanced,
  title = {Enhanced PINO: Production-Ready Physics-Informed Neural Operator},
  author = {Thierry Silvio Claude Soreze},
  year = {2025},
  url = {[https://your-repo.example](https://github.com/TSOR666/Physics-Informed-Neural-Operator)}
}
```

---

## 🗺️ Roadmap
- Time‑dependent physics (heat/wave/transport) with explicit time axis.
- Alternative Laplacians (finite differences) for non‑periodic domains.
- Data‑driven + physics hybrid training utilities.
- Exportable ONNX/TensorRT paths for deployment.

