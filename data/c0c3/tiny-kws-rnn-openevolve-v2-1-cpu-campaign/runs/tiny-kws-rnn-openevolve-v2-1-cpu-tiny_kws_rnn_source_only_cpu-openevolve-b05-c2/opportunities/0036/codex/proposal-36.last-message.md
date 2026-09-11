MECHANISM: Incremental mid-sequence temporal pruning with signed-extrema readout

HYPOTHESIS: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 26 causal frames while reducing total inference MACs from 356,565,760 to approximately 343,421,440.

INTENDED_EDIT: Replace the 65-unit full-resolution model with the qualified 64-unit signed-extrema architecture and omit six distributed interior frames, adding a three-eighths omission to the proven 27-step schedule.

EVIDENCE: The 27-step 64-unit design achieved 86.38% accuracy, 1.38 points above threshold, after every incremental reduction from 32 through 27 steps qualified; removing one additional interior frame is the smallest next structural compute reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)
=======
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 65, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule
>>>>>>> REPLACE