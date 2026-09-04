MECHANISM: Multiplicative bilateral coactivation fusion

HYPOTHESIS: Adding an elementwise interaction between mirrored feature views will exceed 9,328 correct predictions by exposing bilateral coactivations that linear invariant/disagreement fusion cannot represent directly.

INTENDED_EDIT: Add a zero-initialized 64-channel mirrored-feature product stream to the existing fusion layer, preserving the baseline function at initialization.

EVIDENCE: Disagreement-only fusion fell to 9,290 while the full invariant pathway reached 9,328, and generic dense-head widening fell to 9,300; this motivates targeted nonlinear view interaction without removing successful features or materially increasing runtime.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
=======
        self.view_fusion = nn.Conv2d(
            192, 64, kernel_size=1, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
=======
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        bilateral_coactivation = features * flipped_features
        fused = self.view_fusion(
            torch.cat(
                (invariant, disagreement, bilateral_coactivation),
                dim=1,
            )
        )
>>>>>>> REPLACE