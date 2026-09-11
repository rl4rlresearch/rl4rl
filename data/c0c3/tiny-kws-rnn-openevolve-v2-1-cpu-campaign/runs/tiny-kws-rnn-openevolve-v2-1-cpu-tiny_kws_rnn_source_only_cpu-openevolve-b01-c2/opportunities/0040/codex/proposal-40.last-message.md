MECHANISM: Earliest-boundary temporal pruning

HYPOTHESIS: Processing frames 4–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 21 recurrent steps and approximately 247,482,900 total MACs.

INTENDED_EDIT: Adopt the qualified non-affine normalization and bias-free seven-logit head, then remove only frame 3 from the qualified 22-step schedule while preserving continuous frames through 21 and the proven late frames 24, 26, and 28.

EVIDENCE: The 22-step frames 3–21, 24, 26, and 28 design achieved 85.40%; replacing frame 21 with frame 22 missed at 84.66%, and prior removal of frame 26 also missed, motivating preservation of the later contiguous context and all three late observations while pruning the earliest selected boundary frame.

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
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(4, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE