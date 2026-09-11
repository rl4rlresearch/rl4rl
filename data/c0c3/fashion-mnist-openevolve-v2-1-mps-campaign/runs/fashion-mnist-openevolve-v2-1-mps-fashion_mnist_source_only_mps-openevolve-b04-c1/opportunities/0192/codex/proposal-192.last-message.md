MECHANISM: Zero-initialized local spatial refinement

HYPOTHESIS: Adding a residual 7×7 convolutional bottleneck before the proven position-sensitive classifier will exceed 9,330 correct predictions by learning local spatial interactions without enlarging the underperforming dense projection.

INTENDED_EDIT: Add a zero-initialized 64→32→32→64 residual convolutional bottleneck after view fusion, increasing parameters from 224,442 to 237,754 while initially preserving the current function exactly.

EVIDENCE: Widening the spatial classifier fell to 9,301 correct and dual-statistic pooling reached 9,311, while global pooling fell to 9,290; this favors enriching the existing spatial representation with local processing instead of adding dense capacity or changing its pooling statistics.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.spatial_refinement = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                32, 32, kernel_size=3, padding=1, bias=False
            ),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.spatial_refinement[-1].weight)
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
        fused = fused + self.spatial_refinement(fused)
        return self.classifier(fused)
>>>>>>> REPLACE