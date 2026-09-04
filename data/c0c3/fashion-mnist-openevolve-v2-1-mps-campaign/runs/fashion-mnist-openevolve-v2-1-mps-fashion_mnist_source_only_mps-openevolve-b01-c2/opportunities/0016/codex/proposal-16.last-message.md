MECHANISM: Ensemble-dominant paired-flip supervision

HYPOTHESIS: Restoring the best batch-64 residual model and weighting its validation-matched ensemble loss at 75% will exceed 9,312 correct predictions while retaining useful per-view supervision.

INTENDED_EDIT: Remove harmful channel attention, restore batch size 64, and change the paired-flip objective from equal weighting to 75% ensemble loss and 25% individual-view loss.

EVIDENCE: The channel-attention design scored 9,296 versus 9,312 for the ungated batch-64 model; paired-view supervision previously improved 9,261 to 9,276, but its equal weighting with the inference-aligned ensemble loss remains untested.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.GELU(),
            nn.Linear(4, 64),
            nn.Sigmoid(),
        )
=======
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = F.gelu(features + self.refine(features))
        channel_scale = 2.0 * self.channel_gate(features)
        features = features * channel_scale.view(-1, 64, 1, 1)
        return self.classifier(features)
=======
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return 0.5 * (ensemble_loss + view_loss)
=======
    return 0.75 * ensemble_loss + 0.25 * view_loss
>>>>>>> REPLACE