MECHANISM: Content-adaptive spatial recalibration

HYPOTHESIS: Adding a learned spatial gate while preserving the position-sensitive flattened head will exceed 9,210 correct predictions by emphasizing informative foreground locations before classification.

INTENDED_EDIT: Add a zero-initialized 3×3 spatial gate driven by channel-wise mean and maximum maps; apply its bounded residual scale after the existing channel gate. This adds 19 learned parameters.

EVIDENCE: Cardinal augmentation improved correctness from 9,196 to 9,206 and multi-view pooling reached 9,210, indicating unresolved spatial-alignment sensitivity. Global spatial classification fell to 9,128, so this challenges the channel-only recalibration assumption while retaining the successful spatially explicit classifier.

<<<<<<< SEARCH
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.classifier = nn.Sequential(
=======
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_scale = 1.0 + 0.5 * torch.tanh(channel_signal)
        features = features * channel_scale[:, :, None, None]
        return self.classifier(features)
=======
        channel_scale = 1.0 + 0.5 * torch.tanh(channel_signal)
        features = features * channel_scale[:, :, None, None]
        spatial_context = torch.cat(
            (
                features.mean(dim=1, keepdim=True),
                features.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_scale = 1.0 + 0.5 * torch.tanh(
            self.spatial_gate(spatial_context)
        )
        features = features * spatial_scale
        return self.classifier(features)
>>>>>>> REPLACE