# World Model Design Documentation (Part 1)

# 1. Introduction

## 1.1 Motivation

Dynamic object understanding from event cameras is fundamentally different from conventional RGB video segmentation. Unlike frame-based cameras, event cameras only measure asynchronous brightness changes, producing sparse spatio-temporal event streams instead of dense images.

This representation provides several advantages:

- extremely high temporal resolution
- high dynamic range
- low latency
- motion sensitivity
- robustness to illumination changes

However, the lack of texture and appearance information makes semantic understanding significantly more difficult. A moving edge may correspond to a dynamic object, camera ego-motion, illumination changes, sensor noise, or combinations of these effects.

Traditional event-based segmentation methods therefore often rely on:

- optical flow estimation
- supervised pixel annotations
- handcrafted motion compensation
- frame reconstruction

Each of these approaches introduces additional complexity or requires expensive human annotations.

The objective of this work is fundamentally different.

Instead of directly predicting segmentation masks from event streams, the network is designed to first learn an internal latent representation of the surrounding world. Dynamic objects are then identified as regions where the predicted world evolution cannot be fully explained by camera motion.

This shifts the problem from supervised segmentation to self-supervised world modeling.

---

# 2. Overall Design Philosophy

The architecture is built around a simple assumption:

> A model that understands how the static world evolves under ego-motion will naturally expose regions that violate this prediction.

Rather than learning appearance-based object detectors, the network learns:

- scene geometry
- camera motion
- temporal consistency
- latent world evolution

Only after constructing this latent world representation does the model estimate dynamic regions.

Consequently, dynamic segmentation becomes a byproduct of predicting the future state of the environment.

This philosophy closely resembles predictive coding, where unexpected observations correspond to prediction errors.

---

# 3. High-Level Network Overview

The complete architecture consists of four major stages.

```
Event Stream
      │
      ▼
Event Encoder
      │
      ▼
Motion Fusion
      │
      ▼
Temporal Encoder
      │
      ├────────► Depth Head
      │
      ├────────► Pose Head
      │
      ▼
World Transition
      │
      ▼
Latent Renderer
      │
      ▼
Residual Alignment
      │
      ▼
Temporal Memory
      │
      ▼
World Decoder
      │
      ▼
Dynamic Mask Head
```

The network progressively transforms sparse asynchronous events into a dense latent representation describing the surrounding world before finally producing a dynamic object probability map.

Unlike conventional encoder-decoder architectures, the majority of computation occurs in latent space rather than pixel space.

---

# 4. Design Objectives

The architecture was designed according to the following principles.

## 4.1 Modular Design

Each component solves one well-defined problem.

Instead of constructing one monolithic network, the architecture decomposes the task into independent modules.

Advantages include:

- easier debugging
- independent testing
- modular replacement
- straightforward ablation studies
- simpler future extensions

Every module was individually implemented and verified before integration into the complete forward pipeline.

---

## 4.2 Geometry-Aware Learning

Rather than learning arbitrary feature transformations, explicit geometric reasoning is embedded into the architecture.

Examples include:

- depth estimation
- camera pose estimation
- latent feature rendering
- temporal alignment

This introduces strong physical priors that reduce the burden on purely data-driven learning.

---

## 4.3 Self-Supervised Learning

The architecture avoids dependence on manually annotated dynamic masks.

Instead, supervision originates from consistency constraints such as

- temporal consistency
- geometric consistency
- latent reconstruction
- predictive world modeling

Ground-truth masks are used only for evaluation and are not required during training.

---

## 4.4 Latent Space Processing

Most expensive reasoning occurs after the encoder.

Instead of operating on

```
640 × 480
```

features, the network performs reasoning in

```
30 × 40
```

latent space.

Advantages include:

- significantly reduced computational cost
- larger receptive fields
- improved temporal modeling
- easier memory aggregation

Only the final decoder returns to image resolution.

---

# 5. Complete Data Flow

The complete forward pipeline is summarized below.

## Stage 1

Input event voxel grids

```
(B,T,5,H,W)
```

↓

Event Encoder

↓

Multi-scale feature pyramid

```
(B,T,32,240,320)

(B,T,64,120,160)

(B,T,128,60,80)

(B,T,256,30,40)
```

Only the deepest representation is used for world modeling.

---

## Stage 2

IMU measurements

↓

IMU Encoder

↓

Motion embeddings

```
(B,T,128)
```

↓

Motion Fusion

↓

Motion-aware visual features

```
(B,T,256,30,40)
```

---

## Stage 3

Temporal Encoder

↓

Temporal latent sequence

```
(B,T,256,30,40)
```

This representation contains spatio-temporal information across the input history.

---

## Stage 4

Two prediction heads operate independently.

### Depth Head

Produces

```
Depth

(B,1,30,40)
```

### Pose Head

Produces

```
Camera Pose

(B,6)
```

These predictions are later combined to geometrically transform latent features.

---

## Stage 5

World Transition

Input

```
previous latent state

+

motion embedding
```

↓

Predicts

```
future latent state
```

This module models how the static world should evolve under ego-motion.

---

## Stage 6

Latent Renderer

Inputs

- predicted latent state
- predicted depth
- predicted pose
- camera intrinsics
- distortion parameters

↓

Geometrically warps latent features into the coordinate system of the current observation.

Unlike image warping, rendering occurs entirely in feature space.

---

## Stage 7

Residual Alignment

Because rendering cannot perfectly compensate for

- depth errors
- pose errors
- interpolation artifacts
- occlusions

a lightweight residual alignment module learns small feature corrections.

Output

```
(B,T,256,30,40)
```

---

## Stage 8

Temporal Memory

All aligned latent features are aggregated into a single world representation using a Transformer encoder.

Output

```
(B,256,30,40)
```

This tensor represents the network's internal estimate of the current world state.

---

## Stage 9

World Decoder

The compact latent representation is progressively upsampled back to full image resolution.

Output

```
(B,16,480,640)
```

Unlike a traditional U-Net decoder, this stage performs only spatial reconstruction because semantic reasoning has already been completed in latent space.

---

## Stage 10

Dynamic Mask Head

The final lightweight prediction head converts decoded features into

```
Dynamic Probability

(B,1,480,640)
```

Each pixel indicates the estimated likelihood of belonging to a dynamic object.

---

# 6. Why This Pipeline?

A conventional segmentation network attempts to learn

```
Events
      ↓
Segmentation
```

directly.

Our proposed model instead learns

```
Events
      ↓
Latent Geometry
      ↓
Motion Understanding
      ↓
World Dynamics
      ↓
Temporal Consistency
      ↓
Dynamic Objects
```

This decomposition provides significantly stronger inductive bias and aligns more closely with how physical scenes evolve over time.

Instead of asking the network to memorize object appearances, the architecture encourages it to understand the underlying structure of the world and identify regions whose behavior cannot be explained by ego-motion alone.

This philosophy forms the foundation of the proposed self-supervised world model.


# World Model Design Documentation (Part 2)

# 7. Detailed Module Design

This section describes every module of the proposed architecture in detail. Rather than viewing the model as a single deep neural network, it should be understood as a sequence of specialized processing stages. Each stage is responsible for solving one specific sub-problem before passing a richer representation to the next stage.

The overall design intentionally separates perception, geometry estimation, motion reasoning, world prediction, temporal aggregation, and segmentation. This modular organization improves interpretability, debugging, extensibility, and future experimentation.

---

# 7.1 Event Encoder

## Purpose

The event encoder converts raw event voxel grids into compact hierarchical feature representations.

Unlike conventional RGB images, event voxels already contain temporal information through discretized time bins. Therefore, the encoder learns spatial structures while preserving short-term temporal information already embedded inside each voxel grid.

Input

```
(B,T,5,H,W)
```

Output

```
Level 1
(B,T,32,240,320)

Level 2
(B,T,64,120,160)

Level 3
(B,T,128,60,80)

Level 4
(B,T,256,30,40)
```

Only the deepest representation is forwarded into the world model.

---

## Why a Multi-scale Encoder?

Lower layers learn

- edges
- corners
- local event patterns

Intermediate layers learn

- motion structures
- object boundaries

Deep layers capture

- semantic information
- long-range context
- global scene understanding

The feature pyramid is retained because future versions of the decoder may exploit skip connections.

---

## Why Residual Blocks?

Residual learning provides

- improved gradient propagation
- deeper feature extraction
- easier optimization
- reduced degradation problems

Residual blocks have become the standard backbone component in modern vision systems because they enable significantly deeper networks without optimization collapse.

---

## Why Downsample to 30×40?

The original event resolution is

```
640 × 480
```

Processing the complete world representation at full resolution would be computationally expensive.

Instead, semantic reasoning is performed in

```
30 × 40
```

which reduces computation by approximately

```
256×
```

while preserving sufficient spatial structure for downstream prediction.

---

# 7.2 IMU Encoder

## Purpose

The IMU encoder transforms raw inertial measurements into a compact motion embedding.

Input

```
Raw IMU sequence

gyro
accelerometer
timestamps
```

Output

```
(B,T,128)
```

Each temporal frame receives one motion embedding.

---

## Why Use IMU?

Event cameras measure relative brightness changes.

Consequently, event motion originates from

- camera ego-motion
- object motion
- both simultaneously

Without inertial measurements, the network must infer camera motion solely from events.

By introducing IMU data, ego-motion estimation becomes significantly easier, allowing the visual stream to focus on scene understanding rather than recovering camera dynamics.

---

## Why Separate Visual and Inertial Streams?

Vision and IMU possess fundamentally different characteristics.

Visual stream

- dense
- spatial
- high dimensional

IMU stream

- low dimensional
- temporal
- physically meaningful

Learning independent representations before fusion has consistently been shown to outperform concatenating raw measurements.

---

## Output Representation

Each motion embedding summarizes

- instantaneous velocity
- angular velocity
- acceleration
- temporal evolution

The embedding is intentionally compact

```
128 dimensions
```

to prevent it from dominating visual features during fusion.

---

# 7.3 Motion Fusion

## Purpose

Motion Fusion injects ego-motion information into visual latent features.

Inputs

```
Visual Feature

(B,T,256,30,40)

Motion Embedding

(B,T,128)
```

Output

```
(B,T,256,30,40)
```

---

## Why Not Simply Concatenate?

Direct concatenation introduces two issues.

First,

visual features possess spatial dimensions,

whereas motion embeddings do not.

Second,

simple concatenation treats motion as another feature channel rather than a conditioning signal.

Instead,

the IMU embedding is projected into latent feature space and used to modulate visual representations.

This produces motion-aware visual features without increasing spatial dimensionality.

---

## Why Fuse Before Temporal Encoding?

Temporal modeling should understand both

- scene appearance

and

- ego-motion

simultaneously.

If motion were added after temporal reasoning, the temporal encoder would not learn motion-conditioned dynamics.

Therefore, fusion is performed before temporal encoding.

---

# 7.4 Temporal Encoder

## Purpose

The temporal encoder models short-term temporal evolution of latent event features.

Input

```
(B,T,256,30,40)
```

Output

```
(B,T,256,30,40)
```

Unlike Temporal Memory, this module preserves the complete temporal sequence.

---

## Why Temporal Modeling?

Individual event frames contain only instantaneous observations.

Many important cues require observing multiple time steps.

Examples include

- object motion

- camera motion

- temporal consistency

- acceleration

- motion direction

The temporal encoder allows neighboring observations to influence one another before any prediction heads are applied.

---

## Why ConvGRU?

ConvGRU provides several desirable properties.

Unlike standard GRUs,

it preserves spatial layout.

Unlike Transformers,

its computational complexity grows linearly with sequence length.

Unlike ConvLSTMs,

it contains fewer parameters while achieving similar performance.

This makes ConvGRU an efficient first choice for latent temporal modeling.

Future work may replace this module with

- Mamba

- State Space Models

- Temporal Transformers

- Video Vision Transformers

---

# 7.5 Depth Head

## Purpose

The depth head predicts scene geometry from the temporally encoded latent representation.

Input

```
(B,256,30,40)
```

Output

```
(B,1,30,40)
```

Only the most recent temporal feature is used.

---

## Why Predict Depth?

Depth enables geometric reasoning.

Without depth,

camera pose alone cannot determine how scene features should move between viewpoints.

Predicted depth allows latent rendering through projective geometry.

---

## Why Predict Only Current Depth?

The temporal encoder already aggregates previous observations.

Therefore,

the newest latent feature contains the richest estimate of current scene geometry.

Predicting depth from every temporal frame would unnecessarily increase computation without providing additional supervision.

---

# 7.6 Pose Head

## Purpose

The pose head estimates relative camera motion.

Input

```
Latest IMU embedding

(B,128)
```

Output

```
(B,6)
```

Representing

```
(tx, ty, tz)

(rx, ry, rz)
```

---

## Why Estimate Pose from IMU?

The IMU encoder is specifically trained to summarize camera motion.

Using visual features would require the network to disentangle

- appearance

- geometry

- motion

simultaneously.

Instead,

camera motion is estimated directly from the inertial representation, simplifying learning and encouraging specialization.

---

## Why Six Degrees of Freedom?

The renderer requires

- translation

and

- rotation

to transform latent features between viewpoints.

The six-dimensional representation naturally captures full rigid-body motion while remaining lightweight and differentiable.

---

# 7.7 World Transition

## Purpose

The World Transition module predicts how the latent world should evolve under ego-motion.

Inputs

```
Previous latent state

(B,256,30,40)

+

Latest motion embedding

(B,128)
```

Output

```
Predicted latent state

(B,256,30,40)
```

---

## Interpretation

This module represents the dynamics model of the latent world.

Instead of directly observing the future,

it predicts

> "Given my current understanding of the world and the estimated camera motion, what should the next latent state look like?"

This is analogous to a transition function in state-space models.

---

## Why Residual Prediction?

The world changes smoothly between consecutive frames.

Predicting the complete future state from scratch would be unnecessarily difficult.

Instead,

the model predicts

```
Δz
```

and computes

```
zₜ = zₜ₋₁ + Δz
```

Residual prediction has several advantages.

- smoother optimization

- faster convergence

- better stability

- easier learning of small motions

---

## Why Condition on Motion?

The same scene evolves differently depending on camera movement.

A forward translation,

sideways translation,

or rotation produce entirely different observations.

Injecting the motion embedding explicitly allows the transition model to learn motion-conditioned dynamics rather than memorizing arbitrary latent changes.

---

# 7.8 Latent Renderer

## Purpose

The latent renderer performs differentiable geometric warping directly in latent feature space.

Inputs

- predicted latent state

- predicted depth

- predicted pose

- camera intrinsics

- distortion parameters

Output

```
Warped latent feature

(B,256,30,40)
```

---

## Why Render Features Instead of Images?

Rendering RGB images would require

- image reconstruction

- texture synthesis

- photometric losses

These tasks are considerably more difficult than reasoning directly in latent space.

Latent rendering avoids unnecessary image synthesis while preserving geometric consistency.

---

## Advantages

Operating in feature space provides

- lower computational cost

- greater robustness to event sparsity

- stronger semantic representations

- easier optimization

The renderer therefore becomes the geometric bridge connecting predicted world dynamics with observed latent representations.

---

# Summary

At this point in the pipeline, the network has completed all geometry-aware reasoning. It has transformed asynchronous event streams into a motion-conditioned latent world representation, predicted the future world state, estimated scene depth and camera motion, and geometrically rendered the predicted latent features into the current camera frame.

The remaining stages are responsible for correcting rendering imperfections, integrating information over time, reconstructing full-resolution representations, and producing the final dynamic object probability map.

---

# Decoder and Dynamic Mask Prediction

After the latent world representation has been produced, the final stage of the network converts this compact representation back into image space for dense prediction.

Unlike conventional segmentation networks, our objective is **not semantic segmentation**. Instead, the decoder predicts only a **binary dynamic object probability map**.

---

## Decoder

### Purpose

The latent world representation exists at

```
30 × 40
```

while the desired prediction is

```
480 × 640
```

Therefore the decoder performs

- learned upsampling
- spatial refinement
- feature recovery

before the final prediction head.

---

## Inputs

```
(B,256,30,40)
```

---

## Outputs

```
(B,16,480,640)
```

The decoder intentionally compresses the feature dimension from

```
256
↓

16
```

before mask prediction.

This significantly reduces computation while still preserving sufficient information for binary segmentation.

---

## Why Upsample Inside the Decoder?

There were two possible designs.

### Option 1 (Chosen)

Decoder performs

```
30×40
↓

60×80

↓

120×160

↓

240×320

↓

480×640
```

Advantages

- cleaner architecture
- prediction head stays lightweight
- easier supervision
- easier visualization
- modular

---

### Option 2

Decoder outputs

```
30×40
```

and the mask head performs the entire upsampling.

Disadvantages

- prediction head becomes unnecessarily large
- decoder learns almost nothing about image-space reconstruction
- poor modularity

Therefore Option 1 was adopted.

---

# Dynamic Mask Head

The final head predicts

```
Dynamic Probability
```

instead of logits.

Input

```
(B,16,480,640)
```

Output

```
(B,1,480,640)
```

using

```
Sigmoid
```

activation.

The network therefore predicts

```
0
↓

Static

1
↓

Dynamic
```

---

## Why Output Probabilities Instead of Logits?

Initially the design returned logits for compatibility with

```
BCEWithLogitsLoss
```

However, our project does **not** possess binary ground-truth dynamic masks.

Instead, the dynamic supervision will be generated online through self-supervised objectives.

Since downstream losses require actual probabilities rather than logits, returning probabilities directly makes the pipeline considerably cleaner.

---

# Complete Model Pipeline

The final network can be summarized as

```
Event Voxels
      │
      ▼
Event Encoder
      │
      ▼
Motion Fusion
      ▲
      │
 IMU Encoder
      │
      ▼
Temporal Encoder
      │
      ├──────────────► Depth Head
      │
      ├──────────────► World Transition
      │                     ▲
      │                     │
      │                 Motion Embedding
      │
      └──────────────► Pose Head
                            ▲
                            │
                     Motion Embedding

World Transition Output
            │
            ▼
Latent Renderer
            │
            ▼
Alignment Network
            │
            ▼
Temporal Memory
            │
            ▼
World Decoder
            │
            ▼
Dynamic Mask Head
            │
            ▼
Dynamic Object Probability
```

This forms one complete differentiable computational graph.

---

# Outputs of the Current Model

The current implementation returns the following intermediate tensors for inspection and future supervision:

| Name | Shape | Purpose |
|-------|---------|----------|
| event_features | (B,T,256,30,40) | encoder output |
| motion_embeddings | (B,T,128) | IMU representation |
| fused_features | (B,T,256,30,40) | fused latent features |
| temporal_features | (B,T,256,30,40) | ConvLSTM temporal output |
| depth | (B,1,30,40) | predicted depth |
| pose | (B,6) | camera motion |
| predicted_state | (B,256,30,40) | world dynamics prediction |
| rendered_state | (B,256,30,40) | geometry-aware rendered latent |
| aligned_features | (B,T,256,30,40) | corrected temporal sequence |
| world_feature | (B,256,30,40) | memory output |
| decoded_feature | (B,16,480,640) | decoder output |
| mask | (B,1,480,640) | final prediction |

Keeping all intermediate outputs greatly simplifies debugging and visualization during early development.

Once training stabilizes, only the outputs required by the loss functions need to be returned.

---