MECHANISM: Zero-initialized bottleneck spatial refinement

HYPOTHESIS: Adding a lightweight residual 7×7 refinement branch will exceed 9,330 correct predictions by improving local spatial feature interactions while preserving the proven fusion and position-sensitive classifier at initialization.

INTENDED_EDIT: Add an 8,352-parameter bottleneck convolutional branch after view fusion, zero-initialize its final projection, and apply it residually before classification.

EVIDENCE: Dense-head widening reached only 9,300 and replacing the established feature path with deeper 7×7 blocks reached 9,291; this instead preserves the 9,330 design exactly at initialization while spending unused parameter capacity on inexpensive spatial refinement.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.post_fusion = nn.Sequential(
            nn.Conv2d(64, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 64, kernel_size=1, bias=False),
        )
        with torch.no_grad():
            self.post_fusion[-1].weight.zero_()
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
        fused = fused + self.post_fusion(fused)
        return self.classifier(fused)
>>>>>>> REPLACE