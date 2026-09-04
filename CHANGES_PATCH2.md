# U-Net Skip Connections — Patch Set 2

This document explains the U-Net skip connection changes, why they exist,
and what behavior they produce. Apply on top of Patch Set 1.

---

## Files Changed (Patch Set 2)

| # | File | Action | Summary |
|---|------|--------|---------|
| 1 | `src/models/world_model/decoder.py` | MODIFIED | Added U-Net skip connections to `DecoderBlock` and `WorldDecoder` |
| 2 | `src/models/world_model/model.py` | MODIFIED | Extract skips from encoder pyramid, pass to decoder, add `event_pyramid` and `skip_features` to output dict |

No new files. No API breaking changes — `WorldModel.forward(voxel_batch, batch)`
signature is unchanged.

---

## What was wrong before

The encoder produces a 4-level pyramid:

```
l1: (B, T,  32, H/2,  W/2)   ┐
l2: (B, T,  64, H/4,  W/4)   │  DISCARDED
l3: (B, T, 128, H/8,  W/8)   ┘
l4: (B, T, 256, H/16, W/16)  ← only this level was used
```

Only the deepest level (`l4`) was passed to the rest of the pipeline. The
decoder then received a single `(B, 256, H/16, W/16)` tensor and
progressively upsampled it to `(B, 16, H, W)`.

**The problem:** without access to the high-resolution encoder features,
the decoder's mask output was effectively a smoothed-up version of the
residual pseudo-label. Object boundaries were blurry because the only
spatial detail available was what's in the H/16 feature map.

---

## What's fixed

The decoder now receives U-Net skip connections at three resolutions:

```
world_feature (B, 256, H/16, W/16)
    │
    ▼ DecoderBlock(256→128, upsample 2×)  ──┐
    │                                       │ concat
    │                                       ▼
    │                            (B, 256, H/8, W/8)
    │                                       │ conv 3×3
    │                            (B, 128, H/8, W/8)  ◄── skip_l3
    │
    ▼ DecoderBlock(128→64,  upsample 2×)  ──┐
    │                                       │ concat
    │                                       ▼
    │                            (B, 128, H/4, W/4)
    │                                       │ conv 3×3
    │                            (B, 64,  H/4, W/4)   ◄── skip_l2
    │
    ▼ DecoderBlock(64→32,   upsample 2×)  ──┐
    │                                       │ concat
    │                                       ▼
    │                            (B, 64,  H/2, W/2)
    │                                       │ conv 3×3
    │                            (B, 32,  H/2, W/2)   ◄── skip_l1
    │
    ▼ DecoderBlock(32→16,   upsample 2×)
    │                            (B, 16,  H,   W)
    │
    ▼ MaskHead(16→1)
                                 (B, 1,   H,   W)
```

The skips are taken at the reference frame `t = T-1` to be consistent
with `temporal_memory` (which extracts the last token as the
reference-frame representation).

---

## Per-File Changes

### `src/models/world_model/decoder.py`

**`DecoderBlock`** — rewritten to accept a skip tensor:

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        # Upsample (bilinear, no params)
        # Conv 3×3: (in_channels + skip_channels) → out_channels
        # GroupNorm + GELU
        # Conv 3×3: out_channels → out_channels
        # GroupNorm + GELU

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is None and self.skip_channels > 0:
            # Backward compat: use zeros if skip not provided
            skip = torch.zeros(...)
        if x.shape[-2:] != skip.shape[-2:]:
            # Spatial size safety: interpolate if mismatch
            skip = F.interpolate(skip, size=x.shape[-2:], ...)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x); x = self.norm1(x); x = self.act1(x)
        x = self.conv2(x); x = self.norm2(x); x = self.act2(x)
        return x
```

**`WorldDecoder`** — rewritten as `nn.ModuleList` of 4 `DecoderBlock`s:

```python
class WorldDecoder(nn.Module):
    def __init__(self, input_channels=256, output_channels=16,
                 skip_channels=(128, 64, 32)):
        # Stage 1: in=256, skip=128 (l3), out=128
        # Stage 2: in=128, skip=64  (l2), out=64
        # Stage 3: in=64,  skip=32  (l1), out=32
        # Stage 4: in=32,  skip=0,         out=output_channels

    def forward(self, x, skips=None):
        # skips = [l3_skip, l2_skip, l1_skip] or None
        # If None, uses zeros (backward compat)
```

### `src/models/world_model/model.py`

Two changes in `forward`:

1. **Extract skips** at the reference frame:
```python
skip_l3 = event_pyramid[2][:, -1]  # (B, 128, H/8, W/8)
skip_l2 = event_pyramid[1][:, -1]  # (B, 64,  H/4, W/4)
skip_l1 = event_pyramid[0][:, -1]  # (B, 32,  H/2, W/2)

decoded = self.decoder(
    world_feature,
    skips=[skip_l3, skip_l2, skip_l1],
)
```

2. **Return new keys** in output dict for debugging/visualization:
```python
"event_pyramid": event_pyramid,        # list of 4 tensors
"skip_features": [skip_l3, skip_l2, skip_l1],  # list of 3 tensors
```

The existing keys (`event_features`, `decoded_feature`, `mask`, etc.)
are unchanged.

---

## Why this design

### Why take skips at `t = T-1` (the reference frame)?

`world_feature` comes from `temporal_memory(aligned)`, which extracts the
last token (`x[:, -1]`) as the reference-frame representation. For the U-Net
concatenation to be temporally consistent, the skips must be from the same
timestep.

### Why not temporally aggregate the skips too?

Applying `temporal_encoder` (ConvLSTM) + `temporal_memory` (transformer)
to each pyramid level would give temporally-aligned skips, but at 3× the
compute cost. The high-frequency spatial structure (edges) doesn't change
much over T frames, so the last-frame approximation is good enough for
boundary refinement. This is a known limitation; if you observe temporal
inconsistency in the masks, this is the place to upgrade first.

### Why learned skip-consumer convs (not just concat)?

Each `DecoderBlock.conv1` is `Conv2d(in + skip, out, 3×3, padding=1)` —
this lets the network learn how to fuse the upsampled feature with the
skip (e.g., emphasizing edge information from the skip when the upsampled
feature is uncertain).

### Why bilinear upsampling (not transposed conv)?

Bilinear is parameter-free, deterministic, and avoids checkerboard
artifacts that transposed convs can introduce. The subsequent conv layers
handle the refinement.

---

## Backward Compatibility

**`WorldDecoder.forward(x)` without skips still works** — when `skips=None`,
each block fills in zeros for the skip path. This means:

- ✅ Existing code that constructs `WorldDecoder` and calls `decoder(x)`
  continues to run without crashing
- ✅ The output shape is unchanged: `(B, output_channels, H, W)`
- ⚠️ The numerical output WILL differ from before (new conv weights are
  initialized fresh, channel bookkeeping is different)
- ⚠️ Saved checkpoints from before this patch will have "unexpected"
  keys for the new conv1 weights (which now take `in+skip` channels
  instead of `in`). Load with `strict=False` and re-train.

**`WorldModel.forward(voxel_batch, batch)` signature unchanged** — the
U-Net wiring is internal. The output dict gains 2 new keys
(`event_pyramid`, `skip_features`) but no existing keys are removed.

---

## Parameter Count

| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| `WorldDecoder` | ~1.0M | ~1.2M | +200K |
| `WorldModel` total | 15.07M | 15.28M | +0.21M (+1.4%) |

The decoder gets larger because each `conv1` now processes
`in_channels + skip_channels` instead of just `in_channels`. The
increase is small relative to the rest of the model.

---

## Verification Results

All 10 compatibility checks pass (mirroring `notebooks/12_full_model_test.ipynb`):

```
[PASS] instantiation          — 15.28M params, model constructs cleanly
[PASS] forward                — 480×640 input runs without error
[PASS] finite_outputs         — All 16 output tensors finite (no NaN/Inf)
[PASS] shape_verification     — All shapes match notebook 12 expectations:
                                  event_pyramid[0..3]: (B,T,32/64/128/256, H/2/H/4/H/8/H/16, ...)
                                  skip_features[0..2]: (B,128/64/32, H/8/H/4/H/2, ...)
                                  decoded_feature:    (B,16,H,W)
                                  mask:               (B,1,H,W)
[PASS] gradient_check         — 174/174 params receive finite gradients
[PASS] unet_gradients         — 34/34 decoder params + 3/3 skip-consumer convs
                                receive gradients (proving U-Net path is wired)
[PASS] training_step          — Full forward + TotalLoss + backward + step works
                                Total loss=1.90, residual=0.76, all finite
[PASS] backward_compat        — decoder(x) without skips still works (uses zeros)
[PASS] unet_contribution      — Output with vs. without skips differs by 0.53
                                (mean abs), confirming U-Net actually contributes
[PASS] training_stability     — 3 steps: loss 1.70 → 1.49 → 1.35 (decreasing)
```

---

## What's NOT in this patch (deferred)

These remain on the roadmap:

1. **Multi-scale residual supervision** — supervise the mask at multiple
   decoder resolutions, not just the final one. Helps gradients reach
   deeper layers.
2. **Temporal aggregation of skips** — currently takes `t = T-1` only.
   Could apply `temporal_encoder + temporal_memory` to each pyramid level
   (3× compute cost).
3. **4th skip from stem** — the encoder stem produces features at full
   resolution `(B, T, 32, H, W)` which could be used as a 4th skip. Not
   included because the encoder's stem only has 32 channels and isn't very
   informative.
4. **6D rotation representation** — still deferred (axis-angle with grad
   clipping is the current approach).
5. **Edge-aware depth smoothness, motion-compensated mask consistency,
   trainer implementation** — see Patch Set 1's CHANGES.md for full list.
