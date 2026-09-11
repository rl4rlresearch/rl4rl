MECHANISM: Boundary-preserving width–time exchange

HYPOTHESIS: A 69-unit GRU retaining the latest 22 indices of the successful boundary-aware schedule will preserve at least 85% accuracy while reducing total dense inference MACs by approximately 1.8% versus the qualifying 68-unit, 23-step model.

INTENDED_EDIT: Increase recurrent width from 68 to 69, resize the four-statistic classifier from 272 to 276 inputs, and remove the earliest remaining scheduled frame to execute 22 recurrent steps.

EVIDENCE: Removing frame zero from the verified 24-frame pattern allowed the 68-unit, 23-step model to reach 85.52%, whereas uniform 23-frame resampling reached only 83.93%; preserving the successful sample locations while dropping the next earliest boundary frame tests another favorable width–time exchange.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 8)
=======
        self.gru = nn.GRU(20, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(276, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 69), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 23:
            return list(range(available_frames))
        qualifying_schedule = [
            step * (available_frames - 1) // 23
            for step in range(24)
        ]
        return qualifying_schedule[1:]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 22:
            return list(range(available_frames))
        if available_frames == 23:
            return list(range(1, available_frames))
        qualifying_schedule = [
            step * (available_frames - 1) // 23
            for step in range(24)
        ]
        return qualifying_schedule[2:]
>>>>>>> REPLACE