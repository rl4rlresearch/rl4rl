MECHANISM: Nested edge-frame causal subsampling

HYPOTHESIS: A 111-unit GRU using the qualified 29-frame schedule with only its second, edge-adjacent frame removed will retain at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.45%.

INTENDED_EDIT: Reduce the GRU width to 111 and use a 28-step schedule nested within the qualified 29-step schedule by removing frame index 1.

EVIDENCE: The 111-unit, 29-step design qualified at 85.03%, whereas the failed uniform 28-step design changed four interior omissions at once; a nested schedule isolates one additional, likely redundant boundary-frame removal while preserving every selected interior frame from the qualified design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(30, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) > 2:
            del schedule[1]
        return schedule
>>>>>>> REPLACE