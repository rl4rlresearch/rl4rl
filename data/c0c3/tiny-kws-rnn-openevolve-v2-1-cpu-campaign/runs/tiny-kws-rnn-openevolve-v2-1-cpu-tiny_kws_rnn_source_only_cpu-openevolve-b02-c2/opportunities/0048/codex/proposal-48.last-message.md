MECHANISM: One-step-to-one-unit recurrent budget reallocation

HYPOTHESIS: An 85-unit GRU using mel bands 1–17 on frames 5–28 will retain at least 85% validation accuracy while reducing total inference MACs from 519,679,860 to approximately 509,864,000.

INTENDED_EDIT: Adopt the qualified spectral trunk, remove the earliest frame from its 25-step schedule, and increase hidden width from 84 to 85 to compensate for the lost temporal evidence.

EVIDENCE: The 84-unit, 17-band model met the threshold on frames 4–28, and prior temporal comparisons showed retaining later frames was preferable to retaining earlier ones. Adding one hidden unit while removing one step still saves approximately 9.8 million MACs.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
=======
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(1, available_frames - 2))
=======
        return list(range(5, available_frames - 3))
>>>>>>> REPLACE