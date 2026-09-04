# Self-Supervised Loss Functions

## Overview

The proposed world model is trained entirely through self-supervision. No ground-truth annotations are required for depth, camera pose, optical flow, semantic segmentation, or dynamic object masks.

Instead, the network is optimized using multiple consistency objectives that enforce agreement between different components of the model.

Unlike traditional multi-task learning, every prediction produced by the network supervises another prediction. Consequently, no module learns in isolation; instead, the entire system forms a closed optimization loop.

The complete objective is

\[
L_{total}
=
\lambda_1L_{transition}
+
\lambda_2L_{render}
+
\lambda_3L_{align}
+
\lambda_4L_{memory}
+
\lambda_5L_{mask}
+
\lambda_6L_{sparse}
+
\lambda_7L_{tv}
+
\lambda_8L_{temporal}
\]

where each loss supervises a different physical property of the latent world model.

---

# 1. Transition Prediction Loss

## Purpose

The World Transition module predicts the latent representation of the next frame using

- previous latent world representation
- predicted camera motion

The Temporal Encoder already computes the latent representation of the current frame.

Therefore the predicted latent representation should match the encoded latent representation.

---

### Prediction

\[
\hat z_t
=
f_{transition}
(z_{t-1},u_t)
\]

where

- \(z_{t-1}\) is the previous latent representation,
- \(u_t\) is the IMU motion embedding.

---

### Target

The Temporal Encoder provides

\[
z_t
\]

which serves as the self-supervised target.

---

### Loss

\[
L_{transition}
=
\left\|
z_t
-
\hat z_t
\right\|_1
\]

---

### Modules Optimized

- World Transition
- IMU Encoder

---

# 2. Latent Rendering Consistency Loss

## Purpose

The predicted latent state should explain the next observation through camera geometry.

The renderer receives

- predicted latent state
- predicted depth
- predicted pose
- camera intrinsics

and geometrically warps the latent representation into the next frame.

If the estimated depth and pose are correct,

the rendered latent feature should match the encoded latent feature.

---

### Rendering

\[
z_r
=
Render
(
\hat z_t,
D_t,
P_t
)
\]

---

### Target

\[
z_t
\]

obtained from the Temporal Encoder.

---

### Loss

\[
L_{render}
=
\left\|
z_r
-
z_t
\right\|_1
\]

---

### Modules Optimized

- Depth Head
- Pose Head
- Latent Renderer
- World Transition

This loss provides supervision for depth and pose without requiring any external labels.

# 3. Alignment Consistency Loss

## Purpose

Even with accurate depth and pose estimation, rendered latent features are not perfectly aligned due to

- interpolation,
- discretization,
- occlusion,
- depth estimation errors.

The Alignment module predicts a residual correction.

---

### Prediction

\[
z_a
=
Alignment(z_r)
\]

---

### Target

The encoded latent representation

\[
z_t
\]

---

### Loss

\[
L_{align}
=
\left\|
z_a
-
z_t
\right\|_1
\]

---

### Modules Optimized

- Alignment module

Only the Alignment network is updated by this loss.

---

# 4. Memory Reconstruction Loss

## Purpose

Temporal Memory compresses multiple observations into a compact world representation.

A compact representation should still preserve enough information to reconstruct the aligned latent feature.

---

Pipeline

```text
Aligned Features
        │
Temporal Memory
        │
World Feature
        │
Projection
        │
Reconstructed Latent
```

where the projection can simply be a lightweight 1×1 convolution.

---

### Reconstruction

\[
\hat z_a
=
Projection
(
WorldFeature
)
\]

---

### Target

\[
z_a
\]

---

### Loss

\[
L_{memory}
=
\left\|
\hat z_a
-
z_a
\right\|_1
\]

---

### Modules Optimized

- Temporal Memory
- Projection Layer

This loss prevents the temporal bottleneck from discarding useful scene information.

---

# 5. Dynamic Residual Loss

## Purpose

Dynamic objects naturally appear where geometric prediction fails.

Instead of requiring manually annotated dynamic masks, the network generates its own supervision.

---

### Residual

Compute the latent prediction residual

\[
R
=
|z_t-z_r|
\]

Collapse the channel dimension

\[
R_{map}
=
\frac1C
\sum_{c=1}^{C}
R_c
\]

Normalize

\[
R_{pseudo}
=
\frac
{R_{map}-R_{min}}
{R_{max}-R_{min}}
\]

The normalized residual becomes the pseudo-label for the mask head.

---

### Prediction

The decoder predicts

\[
M
\]

where

\[
0\le M\le1
\]

---

### Loss

A simple L1 formulation

\[
L_{mask}
=
\left\|
M
-
R_{pseudo}
\right\|_1
\]

or Binary Cross Entropy

\[
L_{mask}
=
BCE
(
M,
R_{pseudo}
)
\]

Both are fully self-supervised.

---

### Modules Optimized

- World Decoder
- Dynamic Mask Head

The mask therefore emerges directly from regions that cannot be explained by camera motion.



# 6. Sparsity Prior

## Purpose

Only a small fraction of pixels belong to moving objects.

Without regularization, the trivial solution would predict every pixel as dynamic.

To avoid this, the predicted mask is encouraged to remain sparse.

---

### Loss

\[
L_{sparse}
=
\frac1N
\sum_i
M_i
\]

where

\(N\)

is the total number of pixels.

---

### Modules Optimized

- Dynamic Mask Head

---

# 7. Spatial Smoothness Loss

## Purpose

Dynamic objects are spatially continuous.

The predicted mask should therefore avoid isolated noisy activations.

A Total Variation regularizer is employed.

---

### Loss

\[
L_{tv}
=
\sum
|\partial_xM|
+
|\partial_yM|
\]

---

### Modules Optimized

- Dynamic Mask Head

This produces coherent object boundaries while preserving sharp edges.

---

# 8. Temporal Mask Consistency Loss

## Purpose

Predicted masks should remain temporally consistent after compensating for camera motion.

Instead of directly comparing masks between consecutive frames, the previous mask is geometrically warped using the predicted depth and camera pose.

---

Pipeline

```text
Mask(t−1)
      │
Depth(t−1)
      │
Pose(t)
      │
Camera Intrinsics
      │
Latent Renderer
      │
Warped Mask
      │
Compare
      │
Mask(t)
```

The exact same renderer used for latent feature warping is reused for mask warping.

Only the input tensor changes from

```text
(B,256,H,W)
```

to

```text
(B,1,H,W)
```

No additional geometry module is required.

---

### Warping

\[
M_{warp}
=
Render
(
M_{t-1},
D_{t-1},
P_t
)
\]

---

### Loss

\[
L_{temporal}
=
\left\|
M_t
-
M_{warp}
\right\|_1
\]

---

### Modules Optimized

- Depth Head
- Pose Head
- Dynamic Mask Head

This loss encourages temporal consistency while explicitly accounting for ego-motion, making it substantially stronger than simply comparing masks between adjacent frames.

---

# Final Objective

The complete optimization objective is

\[
\boxed{
L_{total}
=
\lambda_1L_{transition}
+
\lambda_2L_{render}
+
\lambda_3L_{align}
+
\lambda_4L_{memory}
+
\lambda_5L_{mask}
+
\lambda_6L_{sparse}
+
\lambda_7L_{tv}
+
\lambda_8L_{temporal}
}
\]

Together, these objectives enable the entire world model—including depth estimation, camera pose estimation, latent dynamics, temporal memory, and dynamic object segmentation—to be trained without any manually annotated supervision. The model learns by enforcing geometric, temporal, and physical consistency within its own latent world representation.