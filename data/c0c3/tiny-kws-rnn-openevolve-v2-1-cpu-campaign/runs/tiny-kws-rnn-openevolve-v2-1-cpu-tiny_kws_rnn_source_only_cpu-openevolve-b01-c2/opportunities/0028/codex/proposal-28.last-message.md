MECHANISM: One-unit recurrent width pruning on the qualified sparse schedule

HYPOTHESIS: A 59-unit GRU using the proven 25-frame endpoint-preserving schedule and seven-logit head will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 285,913,410.

INTENDED_EDIT: Reduce GRU width from 60 to 59, adopt frames 3–26 plus frame 28, and use seven learned relative logits with a fixed zero reference logit.

EVIDENCE: Reference Design 3 achieved 85.77% accuracy with 60 units, 25 steps, and seven logits; its 0.77-point margin motivates the smallest structural width reduction, which saves about 8.5 million MACs without discarding another observed frame.

<<<<<<< SEARCH
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.hidden_size = 59
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
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
        return list(range(3, available_frames - 5)) + [available_frames - 4]
>>>>>>> REPLACE