MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 13% versus the qualified 104-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 96 while preserving the full-frame schedule and established training procedure.

EVIDENCE: The 104-unit full-resolution GRU achieved 85.89% accuracy, while prior 112- and 120-unit reductions also remained qualified; reduced-frame designs failed, so testing the next eight-unit width reduction is the most informative cost-boundary experiment.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE