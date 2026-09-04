MECHANISM: Width-for-time Pareto continuation

HYPOTHESIS: An 83-unit GRU processing 29 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.2% versus the verified 82-unit, 30-step model.

INTENDED_EDIT: Increase the GRU and dual-timescale readout width from 82 to 83 while reducing the causal schedule from 30 to 29 frames.

EVIDENCE: The adjacent diagonal designs—81 units at 31 steps and 82 units at 30 steps—both passed at 85.15% and 85.52%, respectively; continuing that width-for-time trade is the smallest evidence-supported reduction in exact dense MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(164, 8)
=======
        self.gru = nn.GRU(20, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(166, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(30, available_frames)
=======
        steps = min(29, available_frames)
>>>>>>> REPLACE