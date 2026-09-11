MECHANISM: One-unit recurrent-width boundary probe

HYPOTHESIS: A 97-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.9% versus the verified 98-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 98 to 97 while preserving all 32 causal steps and the verified training procedure.

EVIDENCE: The 98-unit model achieved 85.03% accuracy, matching the verified 99- and 100-unit models; this stable accuracy across consecutive widths motivates testing the next one-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE