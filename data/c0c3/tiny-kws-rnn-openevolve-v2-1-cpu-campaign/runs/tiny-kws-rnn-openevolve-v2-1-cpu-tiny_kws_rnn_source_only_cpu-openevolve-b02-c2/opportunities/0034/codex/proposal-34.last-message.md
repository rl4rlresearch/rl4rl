MECHANISM: Shared low-rank mel projection before gated recurrence

HYPOTHESIS: A learned 20-to-16 input projection feeding the qualified 85-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 633,935,525 to approximately 617,391,025.

INTENDED_EDIT: Preserve the 85-unit recurrent state, dual-view classifier, 29-frame schedule, and training procedure, but factor the GRU’s gate inputs through a shared 16-dimensional learned projection.

EVIDENCE: The 85-unit model achieved 86.99% accuracy, while reducing recurrent width to 83 fell below threshold; preserving width 85 while compressing the 20-band input targets input-side GRU MACs without sacrificing the recurrent capacity implicated by that boundary.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 16)
        self.gru = nn.GRU(16, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        projected = self.input_projection(self.input_norm(frame))
        output, hidden = self.gru(
            projected.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        projected = self.input_projection(self.input_norm(frames))
        outputs, hidden = self.gru(
            projected, hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE