MECHANISM: Residual disagreement-only mirror fusion

HYPOTHESIS: Constraining mirror fusion to add a learned disagreement correction onto the invariant features will exceed 9,328 correct predictions by removing a redundant invariant-channel transformation while preserving the winning early-fusion representation.

INTENDED_EDIT: Replace the 128→64 fusion projection with a zero-initialized 64→64 disagreement projection and add its output residually to the invariant feature map.

EVIDENCE: The 224,442-parameter early-fusion model achieved 9,328 correct, while widening it to 249,618 parameters fell to 9,300 and adding global-max features reached only 9,325; this motivates reducing redundant capacity while retaining invariant/disagreement fusion.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
=======
        self.view_fusion = nn.Conv2d(
            64, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
=======
        fused = invariant + self.view_fusion(disagreement)
>>>>>>> REPLACE