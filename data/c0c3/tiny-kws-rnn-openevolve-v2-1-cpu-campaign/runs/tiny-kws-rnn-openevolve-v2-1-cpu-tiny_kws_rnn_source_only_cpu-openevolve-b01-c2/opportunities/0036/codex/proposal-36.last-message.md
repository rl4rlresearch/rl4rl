MECHANISM: Distributed late-frame pruning with endpoint preservation

HYPOTHESIS: A 60-unit GRU processing frames 3–24, 26, and 28 will achieve at least 85% validation accuracy while reducing total inference MACs to approximately 282,690,900.

INTENDED_EDIT: Adopt the qualified non-affine normalization and bias-free seven-logit head, and use a 24-step schedule that retains informative frame 26 while distributing two-frame gaps across the late sequence.

EVIDENCE: The qualified 25-step design reached 85.52%; the 24-step frames 3–25 plus 28 variant narrowly missed at 84.79%. Replacing frame 25 with later frame 26 preserves the proven endpoint and improves late-time coverage without adding MACs.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 1))
=======
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 7)) + [
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE