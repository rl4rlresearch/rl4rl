MECHANISM: Learned low-rank spectral projection

HYPOTHESIS: Projecting all 17 qualified mel bands into 15 learned recurrent features will retain at least 85% validation accuracy while reducing total inference MACs from 519,679,860 to approximately 514,606,485.

INTENDED_EDIT: Add a bias-free 17-to-15 linear spectral projection before the GRU in both recurrent paths and reduce the GRU input width accordingly, preserving the qualified 25-step schedule, 84-unit state, and dual-view classifier.

EVIDENCE: Fixed 16-feature fusion narrowly missed at 84.54%, while discarding a spectral endpoint performed worse; a learned projection can preserve information from every qualified band while structurally reducing recurrent input cost.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.input_norm = nn.LayerNorm(17)
        self.input_projection = nn.Linear(17, 15, bias=False)
        self.gru = nn.GRU(15, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        recurrent_input = self.input_projection(
            self.input_norm(frame[..., 1:18])
        )
        output, hidden = self.gru(
            recurrent_input.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
        )
=======
        recurrent_inputs = self.input_projection(
            self.input_norm(frames[..., 1:18])
        )
        outputs, hidden = self.gru(
            recurrent_inputs, hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE