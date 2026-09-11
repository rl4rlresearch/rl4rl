MECHANISM: Early-neighbor single-frame thinning

HYPOTHESIS: A 127-unit GRU that removes frame 1 from the qualified 28-frame grid will achieve at least 85% validation accuracy while reducing exact inference MACs and recurrent steps below every qualified design.

INTENDED_EDIT: Reduce the GRU and readout width to 127, then use 27 frames by discarding the second scheduled frame while preserving frame 0, both temporal endpoints, and all late frames.

EVIDENCE: The 127-unit/27-step variant dropping frame 0 reached 84.91%, only one validation example below qualification and outperforming removal of the penultimate frame at 84.66%. Removing adjacent frame 1 instead preserves onset-boundary information while targeting the densely sampled early region.

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
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(1)
        return schedule
>>>>>>> REPLACE