MECHANISM: Learned spectral bottleneck with recurrent-width preservation

HYPOTHESIS: Projecting each 20-band frame to 16 learned features while retaining the qualified 64-unit GRU will maintain at least 85% validation accuracy and reduce total inference MACs below the qualified 63-unit design, to approximately 270,032,320.

INTENDED_EDIT: Add a 20-to-16 linear input projection and change the GRU input width to 16, leaving the 64-unit recurrent state, summaries, schedule, and training procedure unchanged.

EVIDENCE: The 63-unit GRU barely qualified at 85.03%, while 62 units failed at 84.17%; this indicates recurrent-state contraction has reached its accuracy boundary and motivates compressing the spectral input while restoring the better-qualified 64-unit state.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 16)
        self.gru = nn.GRU(16, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_projection(self.input_norm(frame)).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_projection(self.input_norm(frames)),
            hidden.transpose(0, 1).contiguous(),
>>>>>>> REPLACE