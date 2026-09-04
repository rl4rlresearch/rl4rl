MECHANISM: Additive class-specific spatial evidence pooling

HYPOTHESIS: A zero-initialized class-specific log-mean-exp evidence branch will exceed 9,206 correct predictions by learning translation-tolerant local class cues while retaining the position-sensitive flattened classifier.

INTENDED_EDIT: Challenge the assumption that every class decision must pass through one shared dense spatial bottleneck: add per-class 1×1 evidence maps with smooth maximum pooling, and reduce the dense width from 160 to 158 to remain below the parameter ceiling.

EVIDENCE: Replacing spatial classification with global context regressed to 9,128, so this preserves the successful flattened head; cardinalized translation augmentation improved correctness to 9,206, making a complementary translation-tolerant class-evidence pathway plausible.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.class_evidence = nn.Conv2d(96, 10, kernel_size=1)
        nn.init.zeros_(self.class_evidence.weight)
        nn.init.zeros_(self.class_evidence.bias)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 158),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(158, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = features * channel_scale[:, :, None, None]
        return self.classifier(features)
=======
        features = features * channel_scale[:, :, None, None]
        dense_logits = self.classifier(features)
        evidence_maps = self.class_evidence(features).flatten(2)
        evidence_logits = 0.5 * (
            torch.logsumexp(2.0 * evidence_maps, dim=2) - math.log(9.0)
        )
        return dense_logits + evidence_logits
>>>>>>> REPLACE