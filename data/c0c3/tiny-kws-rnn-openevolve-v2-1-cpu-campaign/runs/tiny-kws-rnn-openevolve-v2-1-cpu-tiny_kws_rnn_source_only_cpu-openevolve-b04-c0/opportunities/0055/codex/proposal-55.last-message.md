MECHANISM: Redundant-width pruning on the boundary-aware schedule

HYPOTHESIS: Reducing the qualifying 68-unit, 23-step GRU to 67 units will retain at least 85% validation accuracy while lowering total dense inference MACs by approximately 2.6%.

INTENDED_EDIT: Keep the successful 23-frame boundary-aware schedule and four-statistic readout, but reduce recurrent width from 68 to 67 and resize the classifier and states accordingly.

EVIDENCE: The 67-unit GRU already achieved 85.03% accuracy with the full qualifying 24-frame schedule, while removing its earliest boundary frame and using 68 units improved accuracy to 85.52%; this tests whether that added unit was unnecessary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 8)
=======
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(268, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            (batch_size, 68), -1.0, device=device, dtype=dtype
=======
            (batch_size, 67), -1.0, device=device, dtype=dtype
>>>>>>> REPLACE