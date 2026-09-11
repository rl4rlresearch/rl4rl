MECHANISM: Zero-initialized global-context auxiliary classifier

HYPOTHESIS: Adding a class-specific pooled-context logit path will exceed 9,166 correct predictions by complementing the location-sensitive flattened classifier with translation-invariant average/maximum feature evidence.

INTENDED_EDIT: Reduce the zero-initialized channel-gate bottleneck from 24 to 19 units to fund a 96-to-10 global-context classifier, initialized to preserve the current logits exactly; total parameters increase from 249,789 to 249,794.

EVIDENCE: The 9,166-correct adaptive-detail baseline remains unbeaten by filter, loss, and TTA refinements, while more extensive spatial-context designs timed out; this tests complementary contextual classification with negligible added computation.

<<<<<<< SEARCH
        self.channel_gate = nn.Sequential(
            nn.Linear(96, 24),
            nn.GELU(),
            nn.Linear(24, 96),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.classifier = nn.Sequential(
=======
        self.channel_gate = nn.Sequential(
            nn.Linear(96, 19),
            nn.GELU(),
            nn.Linear(19, 96),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.context_classifier = nn.Linear(96, 10)
        nn.init.zeros_(self.context_classifier.weight)
        nn.init.zeros_(self.context_classifier.bias)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = features * channel_scale[:, :, None, None]
        return self.classifier(features)
=======
        features = features * channel_scale[:, :, None, None]
        global_context = 0.5 * (average_context + maximum_context)
        return self.classifier(features) + self.context_classifier(global_context)
>>>>>>> REPLACE