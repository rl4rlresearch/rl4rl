MECHANISM: Identity-initialized global channel recalibration

HYPOTHESIS: Content-dependent channel gating of the fused 7×7 representation will exceed 9,328 correct predictions by emphasizing class-relevant feature channels without the runtime and optimization costs of spatial refinement or attention.

INTENDED_EDIT: Add a 2,128-parameter squeeze-and-excitation gate after mirrored-view fusion, initialized to preserve the current representation exactly at the start of training.

EVIDENCE: Dense-head widening fell to 9,300 correct and global-max evidence reached only 9,325, while spatial refinement and attention timed out; this motivates a lightweight, structurally targeted use of global context instead of more static head capacity or expensive spatial computation.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.channel_gate[3].weight.zero_()
            self.channel_gate[3].bias.zero_()
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
        fused = fused * (2.0 * self.channel_gate(fused))
        return self.classifier(fused)
>>>>>>> REPLACE