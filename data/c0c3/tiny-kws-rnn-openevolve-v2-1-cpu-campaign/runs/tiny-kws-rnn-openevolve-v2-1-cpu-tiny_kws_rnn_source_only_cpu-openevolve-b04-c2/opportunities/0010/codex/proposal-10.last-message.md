MECHANISM: Three-frame leading-edge temporal trim

HYPOTHESIS: A 103-unit GRU processing frames 3–31 will retain at least 85% validation accuracy while reducing execution from 30 to 29 recurrent steps and recurrent MACs from 929,271,150 to approximately 898,295,445.

INTENDED_EDIT: Reduce the GRU width from 112 to the qualified 103 units and omit the first three frames when more than three are available.

EVIDENCE: The 103-unit GRU remained qualified when trimming one leading frame (85.64%) and two leading frames (85.15%); trimming one additional adjacent frame is the smallest untested reduction below the current 30-step frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(3, available_frames))
>>>>>>> REPLACE