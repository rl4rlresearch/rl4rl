MECHANISM: Capacity-for-time recurrent tradeoff

HYPOTHESIS: A 56-unit GRU processing the verified 26-frame window at indices 3–28 will retain at least 85% validation accuracy while reducing total dense inference MACs below the successful 58-unit, 25-step model.

INTENDED_EDIT: Reduce GRU width from 58 to 56 units, resize its recurrent state and classifier, and restore the 26-frame central schedule.

EVIDENCE: The 58-unit 26-step model achieved 86.01% accuracy, while increasing width to 59 could not compensate for reducing the schedule to 24 steps. This indicates temporal context is more valuable than marginal width and motivates exchanging two hidden units for one additional frame; 56 units is the largest integer width that makes this 26-step configuration cheaper than the current 25-step model.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
        self.gru = nn.GRU(20, 56, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 56, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 56, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 56, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(25, available_frames)
=======
        steps = min(26, available_frames)
>>>>>>> REPLACE