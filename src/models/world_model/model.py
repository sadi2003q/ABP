import torch
import torch.nn as nn

from src.models.world_model.event_encoder import EventEncoder
from src.models.world_model.imu_encoder import IMUEncoder
from src.models.world_model.fusion import MotionFusion


from src.models.world_model.temporal_encoder import TemporalEncoder
from src.models.world_model.world_transition import WorldTransition
from src.models.world_model.pose_head import PoseHead
from src.models.world_model.depth_head import DepthHead
from src.models.world_model.latent_renderer import LatentRenderer
from src.models.world_model.alignment import Alignment
from src.models.world_model.temporal_memory import TemporalMemory
from src.models.world_model.decoder import WorldDecoder
from src.models.world_model.mask_head import DynamicMaskHead


class WorldModel(nn.Module):

    def __init__(
        self,
        num_bins: int = 5,
        event_channels: int = 256,
        imu_hidden: int = 64,
        imu_embedding: int = 128,
        decoder_channels: int = 16,
        memory_type: str = "transformer",
    ):
        super().__init__()

        self.event_encoder = EventEncoder(
            input_channels=num_bins,
        )

        self.imu_encoder = IMUEncoder(
            hidden_channels=imu_hidden,
            embedding_dim=imu_embedding,
        )

        self.motion_fusion = MotionFusion(
            event_channels=event_channels,
            imu_dim=imu_embedding,
        )

        self.temporal_encoder = TemporalEncoder(
            input_channels=event_channels,
            hidden_channels=event_channels,
            kernel_size=3,
        )

        self.depth_head = DepthHead(
            input_channels=event_channels,
        )

        self.pose_head = PoseHead(
            input_dim=imu_embedding,
        )

        self.transition = WorldTransition(
            state_channels=event_channels,
            motion_dim=imu_embedding,
        )

        self.renderer = LatentRenderer()

        self.alignment = Alignment(
            channels=event_channels,
        )

        self.temporal_memory = TemporalMemory(
            channels=event_channels,
            memory_type=memory_type,
        )

        self.decoder = WorldDecoder(
            input_channels=event_channels,
            output_channels=decoder_channels,
        )

        self.mask_head = DynamicMaskHead(
            in_channels=decoder_channels,
        )

        # --------------------------------------------------
        # Apply Kaiming (He) / Xavier (Glorot) initialization
        # to all Conv2d and Linear layers.
        #
        # This is CRITICAL for self-supervised learning because:
        # - Default PyTorch init can produce too-large or too-small
        #   weights, causing vanishing/exploding gradients
        # - Kaiming init is optimal for layers followed by ReLU/SiLU/GELU
        #   (preserves variance through the activation)
        # - Xavier init is optimal for layers followed by sigmoid/tanh
        #
        # The mask_head's final bias is set AFTER this (to -3.0) so
        # the mask starts sparse. The pose_head's final bias is set
        # AFTER this (to std=0.1) so pose starts non-zero.
        # --------------------------------------------------
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming/Xavier initialization to all layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming He init for Conv2d (followed by ReLU/SiLU/GELU)
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                # Xavier Glorot init for Linear (followed by SiLU/GELU)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Re-apply special inits that must override the above:
        # 1. Mask head final bias = -3.0 (sparse mask init)
        from src.models.world_model.mask_head import MASK_BIAS_INIT
        mask_final = self.mask_head.head[-1]
        if isinstance(mask_final, nn.Conv2d):
            nn.init.constant_(mask_final.bias, MASK_BIAS_INIT)

        # 2. Pose head final bias = N(0, 0.1) (non-zero pose init)
        pose_final = self.pose_head.network[-1]
        if isinstance(pose_final, nn.Linear):
            nn.init.normal_(pose_final.bias, mean=0.0, std=0.1)


    def forward(self, voxel_batch, batch):
        event_pyramid = self.event_encoder(voxel_batch)

        event_features = event_pyramid[-1]

        motion_embeddings = []

        for frame in batch.frames:

            embedding = self.imu_encoder(
                frame=frame,
                batch_size=voxel_batch.shape[0],
            )

            motion_embeddings.append(embedding)

        motion_embeddings = torch.stack(
            motion_embeddings,
            dim=1,
        )

        fused = self.motion_fusion(
            event_features,
            motion_embeddings,
        )

        temporal_features = self.temporal_encoder(
            fused
        )

        depths = self.depth_head(
            temporal_features
        )

        depth = depths[:, -1]

        poses = self.pose_head(
            motion_embeddings
        )

        pose = poses[:, -1]

        # --------------------------------------------------
        # Temporal prediction: predict the CURRENT frame's
        # latent from the PREVIOUS frame's latent + motion.
        #
        # Previously this fed temporal_features[:, -1] (the
        # current frame) into the transition, which made the
        # target identical to the input -- the transition
        # collapsed to the identity map (zero residual).
        #
        # Now we use temporal_features[:, -2] (the previous
        # frame) so the transition must actually learn temporal
        # dynamics. Requires T >= 2 (guaranteed by the temporal
        # dataset's history_offsets).
        # --------------------------------------------------
        if temporal_features.shape[1] < 2:
            raise ValueError(
                "Temporal prediction requires T >= 2 frames, "
                f"got T={temporal_features.shape[1]}."
            )
        previous_feature = temporal_features[:, -2]

        predicted_state = self.transition(
            previous_feature,
            motion_embeddings[:, -1],
        )

        # --------------------------------------------------
        # Camera intrinsics, scaled to ACTUAL feature-map resolution.
        #
        # The K from the dataset is in ORIGINAL image coordinates
        # (e.g. 480x640). The feature map may be at a different
        # resolution depending on:
        #   1. Encoder downsample (always 16x)
        #   2. Optional image_scale (trainer may pre-downscale input)
        #
        # We compute the ACTUAL downsample by comparing the input
        # voxel size to the feature map size. This is robust to any
        # image_scale and avoids the bug where K was always divided
        # by 16 (wrong when image_scale != 1.0).
        #
        # Also: the LatentRenderer uses grid_sample with
        # align_corners=True, which maps pixel 0 to -1 and pixel W-1
        # to +1 in normalized coordinates. So the effective coordinate
        # system has W pixels spanning [-1, +1].
        # --------------------------------------------------
        # Compute actual downsample from input/feature sizes
        input_h = voxel_batch.shape[-2]  # may be scaled by trainer
        feature_h = event_pyramid[-1].shape[-2]  # always input/16
        input_w = voxel_batch.shape[-1]
        feature_w = event_pyramid[-1].shape[-1]

        # But K is in ORIGINAL coordinates (not scaled input).
        # We need to figure out the original size from K itself.
        # For a standard camera: cx ≈ W/2, cy ≈ H/2
        # So original_W ≈ 2 * K[0,2], original_H ≈ 2 * K[1,2]
        # (We'll use this to compute the correct scale factor)
        K_raw = torch.stack([
            torch.as_tensor(
                k, device=voxel_batch.device, dtype=torch.float32
            )
            for k in batch.frames[-1].camera_intrinsics
        ]).clone()

        # Original image size (inferred from K)
        # cx ≈ W/2 → original_W ≈ 2*cx
        orig_w = 2.0 * K_raw[:, 0, 2]  # (B,)
        orig_h = 2.0 * K_raw[:, 1, 2]  # (B,)

        # Scale factor: original → feature
        # For each sample in batch, the scale may differ (rare but possible)
        scale_x = orig_w / feature_w  # (B,)
        scale_y = orig_h / feature_h  # (B,)

        # Apply per-sample scaling
        K = K_raw.clone()
        K[:, 0, 0] = K[:, 0, 0] / scale_x  # fx
        K[:, 1, 1] = K[:, 1, 1] / scale_y  # fy
        K[:, 0, 2] = K[:, 0, 2] / scale_x  # cx
        K[:, 1, 2] = K[:, 1, 2] / scale_y  # cy

        distortion = torch.stack([
            torch.as_tensor(
                d, device=voxel_batch.device, dtype=torch.float32
            )
            for d in batch.frames[-1].camera_distortion
        ])

        rendered = self.renderer(
            predicted_state,
            depth,
            pose,
            K,
            distortion,
        )

        aligned_input = temporal_features.clone()

        aligned_input[:, -1] = rendered

        aligned = self.alignment(
            aligned_input
        )

        world_feature = self.temporal_memory(
            aligned
        )

        # --------------------------------------------------
        # U-Net skip connections from encoder pyramid.
        # The encoder produces 4 levels (l1, l2, l3, l4).
        # We extract l1, l2, l3 at the reference frame (t=T-1)
        # and pass them to the decoder as skip features.
        # --------------------------------------------------
        skip_l3 = event_pyramid[2][:, -1]  # (B, 128, H/8, W/8)
        skip_l2 = event_pyramid[1][:, -1]  # (B, 64,  H/4, W/4)
        skip_l1 = event_pyramid[0][:, -1]  # (B, 32,  H/2, W/2)

        decoded = self.decoder(
            world_feature,
            skips=[skip_l3, skip_l2, skip_l1],
        )

        # mask is now LOGITS (no sigmoid in DynamicMaskHead).
        # Use torch.sigmoid(outputs["mask"]) for probabilities.
        mask = self.mask_head(
            decoded
        )

        return {

            "event_features": event_features,

            "motion_embeddings": motion_embeddings,

            "fused_features": fused,

            "temporal_features": temporal_features,

            "depth": depth,

            "depths": depths,

            "poses": poses,

            "pose": pose,

            "predicted_state": predicted_state,

            "rendered_state": rendered,

            "aligned_features": aligned,

            "world_feature": world_feature,

            "decoded_feature": decoded,

            "mask": mask,

            # Sigmoid'd mask probabilities (for visualization + metrics)
            # The mask head outputs logits; this is sigmoid(logits) computed
            # inside DynamicResidualLoss and forwarded for convenience.
            # (Will be None if TotalLoss hasn't been called yet, e.g.
            # during inference.)
            "mask_probs": None,

            "K": K,
            "K_original": K_raw,  # unscaled K (for photometric loss)
            "distortion": distortion,

            # U-Net pyramid + skips (for debugging, visualization,
            # and future multi-scale supervision).
            "event_pyramid": event_pyramid,

            "skip_features": [skip_l3, skip_l2, skip_l1],
        }



