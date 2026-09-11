MECHANISM: Joint recurrent-width and causal-step boundary reduction

HYPOTHESIS: A 111-unit GRU using 29 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the qualified 111-unit, 30-step design.

INTENDED_EDIT: Reduce the GRU, summary state, and classifier width from 120 to 111 and reduce the uniform causal schedule from 30 to 29 frames.

EVIDENCE: The 111-unit, 30-step design qualified at 85.40%, while the 128-unit, 29-step design qualified at 85.03%; combining these individually qualified reductions directly tests the next lower-cost boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(30, available_frames)
=======
        steps = min(29, available_frames)
>>>>>>> REPLACE