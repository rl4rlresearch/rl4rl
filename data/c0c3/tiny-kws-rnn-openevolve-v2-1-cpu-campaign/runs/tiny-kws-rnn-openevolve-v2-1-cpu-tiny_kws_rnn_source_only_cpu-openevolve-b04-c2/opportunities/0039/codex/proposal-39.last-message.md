MECHANISM: Minimal width rebound at the 20-step frontier

HYPOTHESIS: An 87-unit GRU trained on frames 4–23 will recover at least one validation example over the 86-unit, 20-step design, reaching at least 85% accuracy while reducing total inference MACs from the qualified 468,621,740 to approximately 455,777,340.

INTENDED_EDIT: Replace the 97-unit GRU and classifier with 87 units and shorten the standard schedule from 21 to 20 contiguous frames, retaining batch size 16 and the established training procedure.

EVIDENCE: The 86-unit, 20-step design missed qualification by only one validation example at 84.91%, while the adjacent 86-unit, 21-step design reached 85.15%; adding one hidden unit is the smallest capacity increase available while preserving the cheaper 20-step execution.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
=======
        self.gru = nn.GRU(20, 87, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(87, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 87, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 87, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
>>>>>>> REPLACE