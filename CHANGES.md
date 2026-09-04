# Self-Supervised Pipeline — Patch Set 1

This document explains every change in this patch set, why it exists, and
what behavior it produces. Apply by copying the files into your repo,
preserving the directory structure.

---

## Files Changed

| # | File | Action | Summary |
|---|------|--------|---------|
| 1 | `src/losses/dynamic_residual_loss.py` | **NEW** | Spatial pseudo-label loss for the mask head |
| 2 | `src/losses/dynamic_mask_regularization_loss.py` | MODIFIED | Default `target_dynamic_ratio` 1.0 → 0.05 |
| 3 | `src/losses/total_loss.py` | MODIFIED | Add residual loss, detach target, `dynamic_ratio_weight` 1.0 → 0.0 |
| 4 | `src/models/world_model/model.py` | MODIFIED | Feed previous frame to transition, scale K by 1/16 |
| 5 | `src/models/world_model/temporal_memory.py` | MODIFIED | Add learned positional encoding |

No other files are touched. The training/inference scripts are unchanged
(they remain empty placeholders, as in the original repo).

---

## Why These Changes — Per-Fix Rationale

### Fix #6 (NEW) — Residual Pseudo-Label Loss

**File:** `src/losses/dynamic_residual_loss.py` (new)

**Problem it solves.** Before this patch, the mask head received
**zero spatial supervision**. The only signals reaching it were global
scalars (`mask.mean()`, sparsity prior, confidence prior). It could not
learn *where* dynamic objects are — only what overall density to output.

**How it works.** The `LatentRenderer` warps the predicted latent using
predicted depth and camera pose:

- For **static background** pixels, the warp correctly aligns the latent
  with the observed current-frame latent → residual is low.
- For **independently moving objects**, the camera-motion-compensated
  warp fails (object motion is not modeled) → residual is high.

We turn this residual into a soft pseudo-label in `[0, 1]` via per-sample
min-max normalization, then train the mask head to predict it via BCE.

**Key design choices:**

1. **The pseudo-label is detached** so gradients flow ONLY into the mask
   head. The mask must predict the residual, not manipulate it (which
   would create a shortcut through depth/pose).

2. **Per-sample min-max normalization** makes the pseudo-label robust
   to the absolute residual magnitude (which shrinks as training
   progresses and the renderer improves). Only the RELATIVE
   high-residual pixels matter.

3. **Resolution handling.** The residual lives at the latent resolution
   `(H/16, W/16)` because the renderer operates there. The mask is at
   full image resolution `(H, W)` because `WorldDecoder` upsamples 16x.
   We bilinearly upsample the pseudo-label to match the mask before BCE,
   then clamp to `[0, 1]` (bilinear can overshoot at sharp transitions).

   This is correct and is the standard SfMLearner recipe: supervise at
   coarse resolution, predict at full resolution, let the decoder refine.
   The pseudo-label is a noisy teacher — sharpness comes from the
   decoder's learned features, not from the teacher.

**Warmup recommendation.** At step 0, depth and pose are random → the
residual is pure noise. Disable this loss for the first ~1 epoch, or
ramp `residual_loss_weight` from 0 → 1.0 over the first 10% of training.

---

### Fix #1 — Target Leakage in LatentConsistencyLoss

**File:** `src/models/world_model/model.py` + `src/losses/total_loss.py`

**Problem it solves.** Before this patch:

```python
# model.py
current_feature = temporal_features[:, -1]         # input to transition
predicted_state = self.transition(current_feature, ...)

# total_loss.py
target_state = outputs["temporal_features"][:, -1]  # SAME tensor!
```

Because `WorldTransition` is residual (`predicted = state + Δ`), the loss
`‖predicted − target‖ = ‖Δ‖` is minimized when `Δ → 0` — i.e. **the
transition learns the identity map**. There is no temporal learning
signal at all.

**Fix.**

1. Feed the **previous** frame's feature to the transition:

   ```python
   previous_feature = temporal_features[:, -2]
   predicted_state = self.transition(previous_feature, ...)
   ```

   Now the transition must actually learn temporal dynamics (predict
   current from previous + motion). A `T >= 2` check is added so the
   model fails loudly if `history_offsets` is too short.

2. **Stop-gradient on the target:**

   ```python
   target_state = outputs["temporal_features"][:, -1].detach()
   ```

   Without `.detach()`, gradients flow into the encoder on BOTH the
   predictor and target branches. The network can trivially minimize
   the loss by collapsing the encoder to a constant output (degenerate
   solution). With `.detach()`, only the predictor branch (transition +
   renderer + depth + pose) receives gradients, which is the correct
   self-supervised setup (cf. BYOL, SimSiam, SfMLearner).

   Note: `target_state` is also reused by the residual loss below
   (which expects a detached target). Do NOT re-detach there.

---

### Fix #2 — Default `target_dynamic_ratio = 0.05`

**File:** `src/losses/dynamic_mask_regularization_loss.py`

**Problem it solves.** The default was `target_dynamic_ratio = 1.0`,
which means the sparsity loss `(mean - 1.0)^2` pushes the mask toward
**all-ones** (every pixel is "dynamic") — the exact opposite of
sparsity. The docstring itself says ρ ≈ 0.05 for indoor scenes (which
is what EVIMO2 is).

**Fix.** Change default to `0.05`:

```python
def __init__(
    self,
    target_dynamic_ratio: float = 0.05,   # was 1.0
    ...
):
```

EVIMO2 is indoor with a few small dynamic objects, so ρ ≈ 0.05 is
appropriate. If you find the mask is too sparse after training,
increase to 0.10; if too dense, decrease to 0.02.

---

### Fix #3 — `dynamic_ratio_weight` defaults to 0

**File:** `src/losses/total_loss.py`

**Problem it solves.** `dynamic_mask["dynamic_ratio"]` is `mask.mean()`
— a **measurement**, not a loss. Adding it to the total loss directly
minimizes `mask.mean()`, pushing every pixel toward 0. Combined with
Fix #2 (which pushes toward ρ=0.05), the two terms fight each other.

**Fix.** Default the weight to 0.0 (rather than removing the code path
entirely, so existing callers that explicitly pass it still work):

```python
def __init__(
    self,
    ...
    dynamic_ratio_weight: float = 0.0,   # was 1.0
    ...
):
```

The `dynamic_ratio` value is still computed and returned in the loss
dict (for logging), it just no longer contributes to the total. If you
want to re-enable it for an experiment, pass `dynamic_ratio_weight=0.1`
or similar — but be aware this directly minimizes `mask.mean()` and
will fight the sparsity prior.

---

### Fix #5 — Scale Intrinsics K to Feature Resolution

**File:** `src/models/world_model/model.py`

**Problem it solves.** `K` was built from `batch.frames[-1].camera_intrinsics`,
which is the **original-image** intrinsics (e.g., for 260×346:
`fx≈180, cx≈173`).

But `LatentRenderer` operates on the feature map at **H/16, W/16**.
Its pixel grid uses `torch.arange(H)` where H is the feature dim
(e.g., 16×21 for 260×346 input).

So the projection `u = fx * X/Z + cx = 180 * X/Z + 173` produces pixel
coordinates in the range [173, ...], but the feature grid is only 21
pixels wide. **Nearly every point projects far out of bounds**, and
`grid_sample(padding_mode="border")` returns near-border values for
everything — the "geometric warp" signal is effectively constant/random.

**Fix.** Scale `fx`, `fy`, `cx`, `cy` by `1/16` (the encoder downsample
factor). `K[2,2] = 1` is unchanged. Also pin `dtype=torch.float32` to
avoid mixed-precision issues when the source numpy array is float64.

```python
ENCODER_DOWNSAMPLE = 16.0
K = torch.stack([
    torch.as_tensor(k, device=..., dtype=torch.float32)
    for k in batch.frames[-1].camera_intrinsics
]).clone()
K[:, 0, 0] = K[:, 0, 0] / ENCODER_DOWNSAMPLE  # fx
K[:, 1, 1] = K[:, 1, 1] / ENCODER_DOWNSAMPLE  # fy
K[:, 0, 2] = K[:, 0, 2] / ENCODER_DOWNSAMPLE  # cx
K[:, 1, 2] = K[:, 1, 2] / ENCODER_DOWNSAMPLE  # cy
```

If you change the encoder depth later, update `ENCODER_DOWNSAMPLE`
accordingly (it equals `2^num_stages`).

---

### Fix (temporal memory) — Learned Positional Encoding

**File:** `src/models/world_model/temporal_memory.py`

**Problem it solves.** `nn.TransformerEncoderLayer` is
**permutation-equivariant** by default — it cannot distinguish which
token is `t=0` vs `t=T-1`. Yet the model extracts `x[:, -1]` as the
"reference frame" representation. Without positional encoding, that
index carries no temporal information for the attention to use.

**Fix.** Add a learned positional encoding (`nn.Parameter` of shape
`(1, max_seq_len, C)`), initialized with `std=0.02` (ViT/BERT
convention):

```python
self.positional_encoding = nn.Parameter(
    torch.randn(1, max_seq_len, channels) * 0.02
)
```

In `forward`, slice the PE to the actual `T` and add it before the
transformer:

```python
pe = self.positional_encoding[:, :T, :]  # (1, T, C)
x = x + pe                                # broadcast over batch
x = self.encoder(x)
world = x[:, -1]  # now meaningful — attention knows which token is "last"
```

**Why learned over sinusoidal?**

1. `T` is small and fixed (typically 4–8 from `history_offsets`),
   so learned PE is cheap and adequate.
2. Learned PE adapts to the actual temporal structure of the data.
3. No risk of frequency mismatch that sinusoidal PE can have at small `T`.

`max_seq_len` defaults to 32, which is large enough for any reasonable
history. The PE is sliced to `T` at runtime, so changing
`history_offsets` later won't break things as long as `T ≤ max_seq_len`.

**Parameter count increase:** `max_seq_len * channels = 32 * 256 = 8,192`
new parameters — negligible.

---

## Verification

The patch set was verified end-to-end with a synthetic-data smoke test.
All 9 checks pass:

```
[PASS] fix_target_leakage      — transition uses temporal_features[:, -2]
[PASS] fix_target_detach       — target_state is .detach()'d in TotalLoss
[PASS] fix_rho_default         — DynamicMaskRegularizationLoss rho = 0.05
[PASS] fix_dynamic_ratio_weight — dynamic_ratio_weight = 0.0 (metric only)
[PASS] fix_K_scale             — K scaled: fx=11.25 (= 180/16)
[PASS] fix_residual_loss       — residual_loss=0.65, contributes to total
[PASS] fix_resolution_mismatch — pseudo_label upsampled to (B,1,H,W)
[PASS] fix_positional_encoding — PE shape=(1,32,256), sliced+added in forward
[PASS] integration             — forward+backward succeeds, all grads finite
```

Parameter count: 173 → 174 (the new `positional_encoding` parameter).
NaN/Inf gradients: 0 / 0.

---

## Compatibility

**No breaking API changes.** All new constructor args have defaults,
so existing callers (`WorldModel()`, `TotalLoss()`, `TemporalMemory()`,
`DynamicMaskRegularizationLoss()`) continue to work without modification.

**Saved checkpoints:** none exist yet (the original `trainer.py` is empty),
so the new `positional_encoding` parameter doesn't cause a "missing keys"
warning on load. If you later add checkpoints from before this patch,
load with `strict=False` to skip the PE parameter.

**Axis-angle rotation:** NOT touched in this patch set. The axis-angle
singularity at θ=0 still exists (gradient magnitude can reach ~1e8). Use
gradient clipping (`clip_grad_norm_(model.parameters(), max_norm=1.0)`)
to keep training stable until you revisit Fix #4 (6D rotation rep).

**IMU encoder numpy/torch mismatch:** NOT touched. The encoder calls
`.unsqueeze()` on what the dataclass declares as `np.ndarray`. If your
data pipeline already produces torch tensors for these fields (or you've
otherwise worked around it), training will work. If not, you'll see an
`AttributeError` at the first forward pass.

---

## Recommended Next Steps (NOT included in this patch set)

These were identified during the audit but are intentionally deferred:

1. **U-Net skip connections** from `event_encoder` pyramid levels 0–2
   to the corresponding decoder stages. Currently the decoder receives
   only the H/16 `world_feature` and has no access to fine spatial
   detail — mask boundaries will be blurry. This is the highest-leverage
   architectural improvement left.

2. **6D rotation representation** (Zhou et al. CVPR'19) replacing
   axis-angle in `LatentRenderer.pose_to_matrix`. Eliminates the θ=0
   gradient singularity without needing gradient clipping.

3. **Edge-aware depth smoothness** using event-voxel gradients as edge
   weights (currently the depth smoothness loss penalizes ALL
   discontinuities, including correct ones at object boundaries).

4. **Motion-compensated mask temporal consistency** (doc §8 — warp
   `M_{t-1}` with `LatentRenderer` before comparing to `M_t`). The
   `MaskTemporalConsistencyLoss` file exists but is dead code (never
   imported by `TotalLoss`) and uses a simple temporal second-derivative
   instead of the documented warping.

5. **Loss warmup schedule** — ramp `residual_loss_weight` from 0 → 1.0
   over the first ~10% of training, because at step 0 the residuals are
   pure noise (depth/pose are random).

6. **Trainer implementation** — `train.py`, `trainer.py`, `inference.py`,
   and `src/utils/metrics.py` are all 0 bytes. Use AMP (bf16),
   `clip_grad_norm_(max_norm=1.0)`, EMA, per-loss TensorBoard logging,
   and `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`.
