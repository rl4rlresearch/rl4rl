MECHANISM: Shared low-rank spectral bottleneck at the qualified temporal frontier

HYPOTHESIS: A learned 20-to-18 projection feeding the 103-unit GRU over frames 4–31 will retain at least 85% accuracy while reducing total inference MACs from 867,991,300 to approximately 862,103,740.

INTENDED_EDIT: Restore the qualified 28-frame schedule and factor the GRU input through the smallest shared spectral bottleneck that reduces exact MACs while preserving recurrent width.

EVIDENCE: The 103-unit, 28-step design achieved 85.40%, whereas reducing recurrent width to 102 missed at 84.54%; preserving the qualified memory width while compressing only the 20-band input is an orthogonal structural reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 18, bias=False)
        self.gru = nn.GRU(18, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self.input_projection(self.input_norm(frame)).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self.input_projection(self.input_norm(frames)),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))
>>>>>>> REPLACE