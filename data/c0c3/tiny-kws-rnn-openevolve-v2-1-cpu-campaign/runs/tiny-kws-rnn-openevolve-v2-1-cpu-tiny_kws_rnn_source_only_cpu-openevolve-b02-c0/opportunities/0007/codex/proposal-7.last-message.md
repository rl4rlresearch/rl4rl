MECHANISM: One-unit recurrent-width boundary reduction

HYPOTHESIS: A 99-unit full-resolution GRU will retain at least 85% validation accuracy while reducing exact recurrent MACs below the verified 100-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 99 while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 100-unit model achieved 85.40% accuracy while the 96-unit model achieved 84.29%; a one-unit reduction is the smallest available probe of the accuracy boundary and preserves full temporal resolution, unlike the unsuccessful 24- and 16-step designs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE