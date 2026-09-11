MECHANISM: Parameter-free mean–final state fusion

HYPOTHESIS: Elementwise addition of the temporal mean and final GRU state will retain at least 85% validation accuracy while eliminating 834,560 classifier MACs and 1,024 learned parameters versus the passing concatenated-readout model.

INTENDED_EDIT: Replace the 256-wide concatenated classifier input with a 128-wide additive fusion of the same two recurrent representations.

EVIDENCE: The 25-step concatenated readout achieved 86.380% accuracy versus 84.540% for mean-only classification, showing that the final state adds useful information; additive fusion preserves access to both representations without the extra dense classifier width.

<<<<<<< SEARCH
        self.classifier = nn.Linear(256, 8)
=======
        self.classifier = nn.Linear(128, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat((temporal_mean, hidden[:, 0, :]), dim=1)
        )
=======
        return self.classifier(temporal_mean + hidden[:, 0, :])
>>>>>>> REPLACE