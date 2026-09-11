MECHANISM: Opposite-edge spectral pruning

HYPOTHESIS: An 84-unit GRU using mel bands 1–17 over the qualified 29-frame schedule will recover at least 85% validation accuracy while matching the 17-band model’s approximately 602,653,380 total inference MACs.

INTENDED_EDIT: Reduce the recurrent input to 17 bands, but preserve band 17 by discarding the lowest-frequency band instead of the third-highest band.

EVIDENCE: Bands 0–17 achieved 85.03%, while bands 0–16 narrowly missed at 84.79%; testing bands 1–17 distinguishes a dimensional-capacity limit from the loss of informative band 17.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 84, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame[..., :18]).unsqueeze(1),
=======
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames[..., :18]), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE