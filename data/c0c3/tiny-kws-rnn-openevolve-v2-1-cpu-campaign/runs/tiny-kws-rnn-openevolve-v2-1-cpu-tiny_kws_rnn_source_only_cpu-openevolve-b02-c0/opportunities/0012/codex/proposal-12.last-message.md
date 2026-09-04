MECHANISM: Second adjacent high-frequency mel-band pooling

HYPOTHESIS: Independently averaging mel bands 16–17 and 18–19 will retain at least 85% validation accuracy while reducing recurrent MACs by another 7,503,705 versus the verified 19-input design.

INTENDED_EDIT: Reduce the GRU input from 19 to 18 features by retaining bands 0–15 and averaging each of the two highest adjacent band pairs; preserve the 99-unit state, 31-step schedule, and classifier.

EVIDENCE: The current 19-feature model achieved 85.52% accuracy after pooling bands 18–19, improving on the uncompressed 31-step model’s 85.03%; this supports another conservative, locality-preserving input reduction rather than reducing recurrent width or readout capacity, both of which previously failed.

<<<<<<< SEARCH
        self.gru = nn.GRU(19, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)

    def _input_features(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_band = normalized[..., 18:].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :18], high_band), dim=-1)
=======
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)

    def _input_features(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_pairs = normalized[..., 16:].reshape(
            *normalized.shape[:-1], 2, 2
        ).mean(dim=-1)
        return torch.cat((normalized[..., :16], high_pairs), dim=-1)
>>>>>>> REPLACE