MECHANISM: Learned spectral bottleneck before recurrence

HYPOTHESIS: Projecting each 20-band frame to 16 learned features while retaining the verified 65-unit, 26-step GRU will preserve at least 85% validation accuracy and reduce total dense inference MACs by approximately 2.8%.

INTENDED_EDIT: Add a 20-to-16 linear input projection and change the GRU input width from 20 to 16, preserving recurrent capacity, temporal summaries, schedule, and training procedure.

EVIDENCE: The 65-unit, 26-step deviation-readout model achieved 86.13% accuracy, whereas reducing recurrent width to 64 at 26 steps failed with 84.05%; this motivates preserving the successful hidden width while reducing the more redundant spectral input dimension.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 16)
        self.gru = nn.GRU(16, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)
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