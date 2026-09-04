MECHANISM: Identity-initialized global channel attention

HYPOTHESIS: A lightweight channel gate will exceed 9,330 correct predictions by adding global feature context while preserving the position-sensitive 7×7 representation and the proven model exactly at initialization.

INTENDED_EDIT: Add a zero-initialized squeeze-and-excitation gate after view fusion, using global pooled context to rescale channels with negligible spatial computation.

EVIDENCE: Global pooling reduced correctness to 9,290, so spatial layout must remain; the zero-initialized spatial refinement timed out, motivating an identity-preserving global-context branch that pools only its gating signal and adds far less computation.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 64, kernel_size=1),
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
        gate = 1.0 + torch.tanh(self.channel_gate(fused))
        return self.classifier(fused * gate)
>>>>>>> REPLACE