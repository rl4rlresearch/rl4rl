MECHANISM: Width-for-time Pareto trade

HYPOTHESIS: An 82-unit GRU processing 30 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.1% versus the verified 81-unit, 31-step model.

INTENDED_EDIT: Increase the GRU and readout width from 81 to 82 while reducing the causal schedule from 31 to 30 frames.

EVIDENCE: The 82-unit, 32-step model reached 86.01% accuracy, while the 81-unit, 31-step model narrowly passed at 85.15%; trading one additional hidden unit for one fewer recurrent step is the smallest unexplored diagonal reduction and still lowers exact dense MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 81, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(162, 8)
=======
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(164, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 81, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 81, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(31, available_frames)
=======
        steps = min(30, available_frames)
>>>>>>> REPLACE