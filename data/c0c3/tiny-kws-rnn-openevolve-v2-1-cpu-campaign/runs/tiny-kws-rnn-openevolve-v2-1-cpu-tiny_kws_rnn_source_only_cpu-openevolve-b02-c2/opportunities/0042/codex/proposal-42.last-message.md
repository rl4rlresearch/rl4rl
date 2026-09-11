MECHANISM: Opposite-edge spectral pruning preserving the informative upper band

HYPOTHESIS: An 84-unit GRU using mel bands 2–17 over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 602,653,380 to approximately 596,697,360.

INTENDED_EDIT: Structurally reduce the recurrent input from 19 to 16 bands, selecting bands 2–17 in both recurrent execution paths while preserving the qualified recurrent width, dual-view classifier, schedule, and training procedure.

EVIDENCE: The 17-band selection 1–17 achieved 86.26% accuracy, whereas 0–16 missed at 84.79%; this indicates retaining upper band 17 and pruning another low-frequency edge band is the best-supported next spectral reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 84, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(16)
        self.gru = nn.GRU(16, 84, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame[..., :19]).unsqueeze(1),
=======
            self.input_norm(frame[..., 2:18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames[..., :19]), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., 2:18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE