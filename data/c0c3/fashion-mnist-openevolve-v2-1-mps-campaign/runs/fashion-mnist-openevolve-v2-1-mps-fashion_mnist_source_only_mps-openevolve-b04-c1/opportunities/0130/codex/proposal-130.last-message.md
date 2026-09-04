MECHANISM: Identity-initialized mean–variance channel routing

HYPOTHESIS: Conditioning spatial feature channels on each image’s global activation means and variances will exceed 9,328 correct predictions by adapting part and texture evidence to the image before spatial classification.

INTENDED_EDIT: Add a lightweight, zero-initialized context gate that computes per-channel spatial moments and uses them to multiplicatively recalibrate the final convolutional features.

EVIDENCE: A direct global-average logit branch fell to 9,290 and dense-head widening reached 9,300, challenging the assumption that global summaries or more static classifier capacity should predict classes directly; using first- and second-order global context to route the successful spatial representation tests a distinct conditional mechanism at negligible computational cost.

<<<<<<< SEARCH
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
=======
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 64),
        )
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
        self.pool = nn.MaxPool2d(2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        return self.pool(features)
=======
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        channel_mean = features.mean(dim=(-2, -1))
        channel_variance = (
            features.square().mean(dim=(-2, -1))
            - channel_mean.square()
        )
        context = torch.cat((channel_mean, channel_variance), dim=1)
        channel_scale = 1.0 + 0.5 * torch.tanh(
            self.context_gate(context)
        )
        features = features * channel_scale[:, :, None, None]
        return self.pool(features)
>>>>>>> REPLACE