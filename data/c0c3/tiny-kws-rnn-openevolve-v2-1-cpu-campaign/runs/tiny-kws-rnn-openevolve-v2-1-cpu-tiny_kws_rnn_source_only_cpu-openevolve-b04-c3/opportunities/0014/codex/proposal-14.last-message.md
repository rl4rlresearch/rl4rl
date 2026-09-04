MECHANISM: Re-optimized 27-frame uniform striding with boundary-width GRU

HYPOTHESIS: A 127-unit GRU using a directly generated 27-frame uniform grid will achieve at least 85% validation accuracy while reducing recurrent steps and exact inference MACs below every qualified design.

INTENDED_EDIT: Reduce the recurrent and classifier width from 128 to 127, and replace the 28-frame grid with a uniformly distributed 27-frame schedule spanning both endpoints.

EVIDENCE: The 127-unit/28-step model qualified at 85.28%, while 127-unit/27-step schedules formed by deleting frames from the 28-frame grid missed by as little as one validation example; regenerating the grid for exactly 27 steps tests whether balanced temporal allocation recovers that narrow margin.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(-2)
        return schedule
=======
        steps = min(27, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
>>>>>>> REPLACE