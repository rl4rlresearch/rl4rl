MECHANISM: Width-for-time Pareto continuation

HYPOTHESIS: An 84-unit GRU processing 28 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.3% versus the verified 83-unit, 29-step model.

INTENDED_EDIT: Increase the GRU and dual-timescale readout width from 83 to 84 while reducing the causal schedule from 29 to 28 frames.

EVIDENCE: The adjacent diagonal designs—81 units at 31 steps, 82 units at 30 steps, and 83 units at 29 steps—all passed; the latest reached 86.01% accuracy, supporting one more incremental width-for-time trade.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(166, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(29, available_frames)
=======
        steps = min(28, available_frames)
>>>>>>> REPLACE