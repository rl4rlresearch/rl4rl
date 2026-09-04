MECHANISM: Anchored residual disagreement fusion

HYPOTHESIS: Fixing the invariant feature path as an identity shortcut while learning only a zero-initialized disagreement correction will exceed 9,330 correct predictions by reducing redundant projection drift without sacrificing flip-specific information.

INTENDED_EDIT: Change `view_fusion` to project only the 64-channel disagreement tensor, then add its output residually to the invariant features.

EVIDENCE: Increasing the EMA rate of `view_fusion` reduced correctness from 9,330 to 9,326, suggesting that changes to the invariant projection are harmful; anchoring that path preserves the proven spatial representation while retaining a learnable correction.

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