MECHANISM: Overlapping adjacent-band spectral compression

HYPOTHESIS: Compressing 20 normalized mel bands into 19 overlapping adjacent-band averages will preserve at least 85% validation accuracy while reducing exact recurrent MACs below the qualified 128-unit, 27-step design.

INTENDED_EDIT: Keep the qualified 27-frame schedule and full 128-unit recurrent/readout capacity, but reduce the GRU input width from 20 to 19 using a parameter-free adjacent-band average in both recurrent execution paths.

EVIDENCE: The current 128-unit, 27-step model qualified at 85.15%, while trimming one classifier feature failed at 83.80%; this instead preserves the entire recurrent state and classifier while removing only the highest-frequency spectral difference mode.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(19, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        hidden, summary, count = state
        normalized = self.input_norm(frame)
        compressed = 0.5 * (normalized[..., :-1] + normalized[..., 1:])
        output, hidden = self.gru(
            compressed.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        hidden, summary, count = state
        normalized = self.input_norm(frames)
        compressed = 0.5 * (normalized[..., :-1] + normalized[..., 1:])
        outputs, hidden = self.gru(
            compressed, hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE