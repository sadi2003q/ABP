"""

Pose Temporal Consistency Loss.

This loss regularizes the sequence of predicted relative camera poses.

Rather than penalizing motion directly, it encourages consecutive
relative pose estimates to vary smoothly over time, preventing
physically implausible discontinuities while allowing legitimate
camera motion.

No ground-truth pose is required.

Input
-----
poses

    Sequence of predicted relative poses.

    Shape:
        (B,T,6)

Output
------
loss

    Scalar temporal consistency loss.

    
Pose representation

-------------------



Each pose is represented as



    (tx, ty, tz, rx, ry, rz)



where



    t : translation



    r : rotation



Shape

-----



Input



    poses



        (B,T,6)



Output



    Scalar loss.



Example usage

-------------



outputs = model(

    voxel_batch,

    batch,

)



#

# Assume poses are predicted for every timestep

#



pose_loss = pose_smoothness_loss(



    outputs["pose_sequence"]



)



loss += pose_loss

"""



from __future__ import annotations



import torch

import torch.nn as nn

import torch.nn.functional as F





class PoseTemporalConsistencyLoss(nn.Module):

    """

    Temporal pose smoothness regularizer.

    """



    def __init__(

        self,

        beta: float = 1.0,

    ):

        super().__init__()



        self.beta = beta



    def forward(

        self,

        poses: torch.Tensor,

    ) -> dict[str, torch.Tensor]:

        """

        Parameters

        ----------

        poses



            Predicted camera poses.



            Shape



                (B,T,6)



        Returns

        -------

        torch.Tensor



            Scalar smoothness loss.

        """



        if poses.ndim != 3:



            raise ValueError(

                "poses must have shape (B,T,6)"

            )



        if poses.shape[-1] not in (6, 9):



            raise ValueError(

                "Last dimension must be 6 (axis-angle) or 9 (6D rotation)."

            )



        #

        # Frame-to-frame pose difference

        #



        delta_pose = (



            poses[:, 1:]



            -



            poses[:, :-1]



        )



        #

        # Robust smoothness penalty

        #



        loss = F.smooth_l1_loss(



            delta_pose,



            torch.zeros_like(delta_pose),



            beta=self.beta,



        )



        return {
            "loss": loss
        }