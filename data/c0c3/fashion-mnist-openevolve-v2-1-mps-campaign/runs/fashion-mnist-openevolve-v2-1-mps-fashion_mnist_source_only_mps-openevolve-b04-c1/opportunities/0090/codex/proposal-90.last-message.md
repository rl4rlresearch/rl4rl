MECHANISM: Reflection-context channel recalibration

HYPOTHESIS: Image-conditioned multiplicative channel gating will exceed 9,328 correct predictions by letting global invariant and disagreement evidence determine which fused channels matter for each image.

INTENDED_EDIT: Add a zero-initialized squeeze-excitation branch over the paired mirror representation and use it to recalibrate the existing fused feature map while preserving baseline behavior at initialization.

EVIDENCE: The static early-fusion model remains best at 9,328, while widening its classifier fell to 9,300 and cached late pooling fell to 9,218; this motivates retaining early fusion while replacing generic capacity with targeted nonlinear interactions between global reflection context and spatial features.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.view_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.Tanh(),
        )
        with torch.no_grad():
            self.view_context[-2].weight.zero_()
            self.view_context[-2].bias.zero_()
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        paired_context = torch.cat((invariant, disagreement), dim=1)
        fused = self.view_fusion(paired_context)
        channel_scale = 1.0 + 0.5 * self.view_context(paired_context)
        fused = fused * channel_scale[:, :, None, None]
        return self.classifier(fused)
>>>>>>> REPLACE