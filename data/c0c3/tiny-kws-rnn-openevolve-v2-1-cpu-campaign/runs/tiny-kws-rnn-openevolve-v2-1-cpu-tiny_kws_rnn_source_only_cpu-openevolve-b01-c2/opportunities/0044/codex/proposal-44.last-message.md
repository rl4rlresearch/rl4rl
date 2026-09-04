MECHANISM: Minimal recurrent-width contraction

HYPOTHESIS: A 59-unit GRU using the qualified 22-frame schedule will retain at least 85% validation accuracy while reducing total dense inference MACs from 259,218,900 to approximately 251,724,975.

INTENDED_EDIT: Adopt the qualified non-affine normalization, bias-free seven-logit head, and frames 3–21, 24, 26, and 28 schedule, while reducing GRU width from 60 to 59 units.

EVIDENCE: The 60-unit 22-step design achieved 85.40% accuracy, while tested 21-step schedules failed; preserving all qualified observations and making the smallest possible recurrent-width reduction targets MACs without discarding additional temporal or spectral information.

<<<<<<< SEARCH
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.hidden_size = 59
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
=======
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE