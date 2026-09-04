MECHANISM: Adjacent temporal edge pruning with the qualified spectral trunk

HYPOTHESIS: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 29 to 28 causal steps, lowering total inference MACs from 602,653,380 to approximately 581,910,000.

INTENDED_EDIT: Adopt the qualified 17-band input selection and omit one additional trailing frame while preserving recurrent width, dual-view classification, and training procedure.

EVIDENCE: The 17-band bands-1–17 design achieved 86.26% accuracy at 602,653,380 MACs, while reducing to 16 bands failed; preserving its spectral and recurrent capacity while testing the nearest temporal reduction isolates a new cost axis with substantially larger potential savings.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
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
        return list(range(1, available_frames - 3))
>>>>>>> REPLACE