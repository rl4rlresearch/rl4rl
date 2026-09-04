MECHANISM: Affine-folded input normalization with reference-class logits

HYPOTHESIS: Removing LayerNorm’s foldable affine parameters while adopting the qualified 25-step, seven-logit design will retain at least 85% validation accuracy at 294,426,900 MACs and reduce parameters from 16,067 to 16,027.

INTENDED_EDIT: Process frames 3–26 and frame 28, learn seven relative logits with a fixed zero reference logit, and disable the redundant affine transform in the input LayerNorm.

EVIDENCE: Reference Design 2 achieved 85.77% accuracy at 294,426,900 MACs over 25 steps. LayerNorm’s default affine transform is followed immediately by the GRU’s learned affine input maps, so its scale and offset are structurally absorbable without reducing model expressivity.

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
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 2))
=======
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
>>>>>>> REPLACE