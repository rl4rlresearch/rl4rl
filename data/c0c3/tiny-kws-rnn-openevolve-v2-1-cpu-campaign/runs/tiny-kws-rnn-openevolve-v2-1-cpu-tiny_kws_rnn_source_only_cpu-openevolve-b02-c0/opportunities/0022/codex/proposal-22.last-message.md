MECHANISM: Single-channel latent recurrent memory

HYPOTHESIS: Restricting classification to 98 of the 99 recurrent channels will retain at least 85% validation accuracy while reducing classifier inference by 6,520 MACs; the unobserved channel remains available as recurrent memory that can influence later observed outputs.

INTENDED_EDIT: Preserve the verified 18-feature, 99-unit, 30-step GRU, but reduce the classifier input from 99 to 98 channels and exclude one summary channel from the final readout.

EVIDENCE: Grouping the readout down to 33 features failed at 82.45%, while reducing the recurrent state itself to 98 units also failed; this conservative change preserves all 99 recurrent units and removes only one of 99 direct classifier inputs.

<<<<<<< SEARCH
        self.classifier = nn.Linear(99, 8)
=======
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(summary / count.clamp_min(1.0))
=======
        pooled = summary / count.clamp_min(1.0)
        return self.classifier(pooled[..., :98])
>>>>>>> REPLACE