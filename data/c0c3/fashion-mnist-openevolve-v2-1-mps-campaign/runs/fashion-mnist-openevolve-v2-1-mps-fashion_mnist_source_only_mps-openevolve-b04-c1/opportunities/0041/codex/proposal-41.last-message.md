MECHANISM: Identity-initialized nonlinear view-fusion correction

HYPOTHESIS: A nonlinear residual fusion of invariant and disagreement features will exceed 9,316 correct predictions by modeling interactions that the current single linear projection cannot represent.

INTENDED_EDIT: Replace the linear 128-to-64 fusion with a 96-channel GELU bottleneck whose zero-initialized output learns corrections to the invariant features, remaining under the parameter ceiling.

EVIDENCE: Joint invariant–disagreement fusion improved validation correct from 9,286 to 9,316, while widening the general classifier regressed; this motivates adding capacity specifically to the successful fusion mechanism.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
=======
        self.view_fusion = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
        )
        with torch.no_grad():
            self.view_fusion[-1].weight.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
=======
        fused = invariant + self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
>>>>>>> REPLACE