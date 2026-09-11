MECHANISM: Residual disagreement-only view fusion

HYPOTHESIS: Removing the redundant invariant-feature transform while retaining a zero-initialized disagreement correction will exceed 9,316 correct predictions by improving optimization within the fixed exposure budget.

INTENDED_EDIT: Replace the 128-to-64 joint projection with a 64-to-64 disagreement projection added residually to the invariant mean, reducing parameters and computation while preserving the initial function.

EVIDENCE: Joint invariant–disagreement fusion improved validation correct from 9,286 to 9,316, whereas added coactivation fell to 9,298. The previous residual-only verification timed out without measuring accuracy; its lower computational cost makes it the most informative unresolved fusion simplification to verify.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
=======
        self.disagreement_fusion = nn.Conv2d(
            64, 64, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.disagreement_fusion.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
=======
        fused = invariant + self.disagreement_fusion(disagreement)
>>>>>>> REPLACE