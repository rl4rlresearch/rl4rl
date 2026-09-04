MECHANISM: Residual disagreement-only view fusion

HYPOTHESIS: Removing the redundant learned transform of invariant features while retaining a zero-initialized disagreement correction will exceed 9,316 correct predictions by simplifying optimization under the fixed exposure budget.

INTENDED_EDIT: Replace the 128-to-64 joint projection with a zero-initialized 64-to-64 disagreement projection added residually to the invariant mean, reducing parameters and computation while preserving the initial function.

EVIDENCE: Mean-and-disagreement fusion improved validation correct from 9,286 to 9,316, whereas added coactivation capacity fell to 9,298; this isolates the successful disagreement mechanism and removes an invariant transform that is expressively redundant with the following linear classifier.

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