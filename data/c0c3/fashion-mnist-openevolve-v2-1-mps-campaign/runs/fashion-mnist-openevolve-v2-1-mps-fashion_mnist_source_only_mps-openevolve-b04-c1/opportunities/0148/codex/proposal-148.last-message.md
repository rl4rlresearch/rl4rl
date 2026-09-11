MECHANISM: Low-cost depthwise-separable spatial residual refinement

HYPOTHESIS: A 4,928-parameter near-identity spatial refinement branch will exceed 9,328 correct predictions by learning local part relationships without the runtime cost of the timed-out larger bottleneck.

INTENDED_EDIT: Add a depthwise-separable residual block after view fusion, initialized at 0.1 output strength; total learned parameters become 229,370.

EVIDENCE: Dense-head widening reached 9,300 and global pooling reached 9,290, while the larger spatial bottleneck timed out; this tests the remaining spatial-processing hypothesis with substantially less computation.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                groups=64,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        with torch.no_grad():
            self.spatial_refine[-1].weight.fill_(0.1)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        fused = fused + self.spatial_refine(fused)
        return self.classifier(fused)
>>>>>>> REPLACE