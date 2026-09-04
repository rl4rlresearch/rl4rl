MECHANISM: One-unit recurrent-width boundary reduction

HYPOTHESIS: A 98-unit full-resolution GRU will retain at least 85% validation accuracy while reducing exact recurrent and classifier MACs below the verified 99-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 99 to 98 while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 99-unit model achieved 85.40% accuracy, while the 96-unit model achieved 84.29%; testing 98 units is the most informative next probe of the minimum viable width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE