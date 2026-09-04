MECHANISM: Recurrent-width compression with full temporal coverage

HYPOTHESIS: A 112-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by 12% versus the verified 120-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 120 to 112 while preserving all 32 causal steps and the training procedure.

EVIDENCE: The 120-unit full-frame model achieved 86.75% accuracy, while 24- and 28-frame models failed; this supports preserving temporal coverage and testing a modest 6.7% reduction in state capacity.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE